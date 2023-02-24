## ----extract-code-for-debug, eval = FALSE--------------------------------------------------------------------------------------------------------------------------------
## knitr::purl(input = "tp_model_entrainment.Rmd", output = "tp_model_entrainment.R")


## ----setup, echo = FALSE, include=FALSE----------------------------------------------------------------------------------------------------------------------------------
rm (list=ls())

#load("~/Experiments/TP_model/tp_model.RData")

#options (digits = 3)
knitr::opts_chunk$set(
    # Run the chunk
    eval = TRUE,
    # Don't include source code
    echo = FALSE, 
    # Print warnings to console rather than the output file
    warning = FALSE,  
    # Stop on errors
    error = FALSE,
    # Print message to console rather than the output file
    message = FALSE,
    # Include chunk output into output
    include = TRUE,
    # Don't reformat R code
    tidy = FALSE,
    # Center images
    # Breaks showing figures side by side, so switch this to default
    fig.align = 'center', 
    # Show figures where they are produced
    fig.keep = 'asis',
    # Prefix for references like \ref{fig:chunk_name}
    fig.lp = 'fig',
    # For double figures, and doesn't hurt for single figures 
    fig.show = 'hold', 
    # Default image width
    out.width = '100%')

# other knits options are here:
# https://yihui.name/knitr/options/



## ----load-libraries, echo = FALSE, include = FALSE, message = FALSE, warning = FALSE-------------------------------------------------------------------------------------

# Read in a random collection of custom functions
if (Sys.info()[["user"]] %in% c("ansgar", "endress")){
    source ("/Users/endress/R.ansgar/ansgarlib/R/tt.R")
    source ("/Users/endress/R.ansgar/ansgarlib/R/null.R")
    #source ("helper_functions.R")
} else {
    # Note that these will probably not be the latest versions
    source("http://endress.org/progs/tt.R")
    source("http://endress.org/progs/null.R")
}

library ("knitr")
library(latex2exp)
library (cowplot)


## ----set-default-parameters-network, echo = FALSE, include = FALSE, message = FALSE, warning = FALSE---------------------------------------------------------------------

# Number of neurons
N_NEURONS <- 19

ACT_FNC <- 'rational_logistic'

# Forgetting for activation
L_ACT_DEFAULT <- 0.4
L_ACT_SAMPLES <- seq (0, 1, .2)
L_ACT <- L_ACT_DEFAULT
#L_ACT <- L_ACT_SAMPLES

# Forgetting for weights
L_W <- 0

# Activation coefficient
A <- .7

# Inhibition coefficient 
B <- .4

# Learning coefficient
R <- 0.05

# noise for activation
NOISE_SD_ACT <- 0.001

# noise for weights
NOISE_SD_W <- 0


## ----set-default-parameters-simulations, echo = FALSE, include = FALSE, message = FALSE, warning = FALSE-----------------------------------------------------------------

# Number of items (e.g., words)
N_WORDS <- 4

# Number of units per item (e.g., syllables)
N_SYLL_PER_WORD <- 3

# Number of repetitions per word
N_REP_PER_WORD <- 100

# Number of repetitions per word before spectral information is computed
N_REP_PER_WORD_BURNIN <- 50

# Number of simulations/subjects
N_SIM <- 100

# Adjust number of neurons if required
if (N_NEURONS < ((N_WORDS * N_SYLL_PER_WORD) + 1))
    N_NEURONS <- (N_WORDS * N_SYLL_PER_WORD) + 1

PRINT.INDIVIDUAL.PDFS <- TRUE
current.plot.name <- "xxx"

# Set seed to Cesar's birthday
set.seed (1207100)


## ----list-parameters, echo = FALSE, results='hide'-----------------------------------------------------------------------------------------------------------------------
list_parameters(accepted_classes = c("numeric")) %>%
    knitr::kable(
        "latex", 
        booktabs = T, 
        caption='\\label{tab:params}Parameters used in the simulations') %>%
    kableExtra::kable_styling()


## ----define-functions, echo = FALSE, include = FALSE, message = FALSE, warning = FALSE-----------------------------------------------------------------------------------

act_fnc <- function (act, fnc = ACT_FNC, ...){
    
    switch (fnc,
            "rational_logistic" = act / (1 + act),
            "relu" = pmax (0, act),
            "tanh" = tanh (act),
            stop ("Unknown activation function"))
}

make_act_vector <- function (ind, n_neurons){
    
    act <- rep (0, n_neurons)
    act[ind] <- 1
    
    return (act)
    
}

update_activation <- function (act, w, ext_input, l_act = 1, a = 1, b = 0, noise_sd = 0, ...){
    # activation, weights, external_input, decay, activation coefficient, inhibition coefficient
    
    act_output <- act_fnc (act, ...)
    
    act_new <- act
    
    # Decay     
    if (l_act>0)
        act_new <- act_new - l_act * act 
    
    # External input
    act_new <- act_new + ext_input
    
    # Excitation
    act_new <- act_new + (a * w %*% act_output)
    
    # Inhibition (excluding self-inhibition)
    act_new <- act_new - (b * (sum (act_output) - act_output))
    
    # Noise
    if (noise_sd > 0)    
        act_new <- act_new + rnorm (length(act_new), 0, noise_sd)
    
    act_new <- as.vector(act_new)
    
    act_new[act_new < 0] <- 0
    
    return (act_new)
}

update_weights <- function (w, act, r = 1, l = 0, noise_sd, ...){
    
    act_output <- act_fnc (act, ...)
    
    # learning 
    w_new <- w  + r * outer(act_output, act_output)
    
    # decay
    if (l > 0)
        w_new <- w_new - l * w 
    
    if (noise_sd > 0)
        w_new <- w_new + as.matrix (rnorm (length(w_new),
                                           0,
                                           noise_sd),
                                    ncol = ncol (w_new))
    
    # No self-excitation
    diag (w_new) <- 0
    
    w_new[w_new < 0] <- 0
    
    return (w_new)
}

familiarize <- function (stream_matrix,
                         l_act = 1,
                         a = 1,
                         b = 0, 
                         noise_sd_act = 0,
                         r = 1,
                         l_w = 0,
                         noise_sd_w = 0,
                         n_neurons = max (stream),
                         return.act.and.weights = FALSE,
                         ...){
    
    # Initialization
    act <- abs(rnorm (n_neurons, 0, noise_sd_act))
    w <- matrix (abs(rnorm (n_neurons^2, 0, noise_sd_w)), 
                 ncol = n_neurons)
    diag(w) <- 0
    
    if (return.act.and.weights)
        act.weight.list <- list ()
    
    # Randomize familiarization 
    stream_matrix <- stream_matrix[sample(nrow(stream_matrix)),]
    # c() conccatenates columns, so this is correct
    stream <- c(t(stream_matrix))
    
    act_sum <- c()
    xxx <- 0
    for (item in stream){
        
        
        xxx <- xxx + 1 
        if ((xxx %% 120) == 1){
            print ("10 rep")
        }
        
        
        current_input <- make_act_vector(item, n_neurons)
        
        act <- update_activation(act, w, current_input, 
                                 l_act, a, b, noise_sd_act,
                                 ...)
        
        if (r > 0)
            w <- update_weights (w, act, r, l_w, noise_sd_w)
        
        act_sum <- c(act_sum, sum(act))
        
        if (return.act.and.weights){
            act.weight.list[[1 + length(act.weight.list)]] <- 
                list (item = item,
                      act = act,
                      w = w)
            
        }
    }
    
    if (return.act.and.weights)
        return (list (
            w = w,
            act_sum = act_sum,
            act.weight.list = act.weight.list))
    else
        return (list (
            w = w,
            act_sum = act_sum))
}

test_list <- function (test_item_list,
                       w,
                       l_act = 1, a = 1, b = 0, 
                       noise_sd_act = 0,
                       n_neurons,
                       return.global.act = FALSE,
                       ...) {
    # Arguments
    #   test_item_list  List of test-items (i.e., numeric vectors)
    #   w               Current weight matrix
    #   l_act           Forgetting rate for activation. Default:  1
    #   a               Excitatory coefficient. Default: 1
    #   b               Inhibitory coefficient. Default: 0
    #   noise_sd_act    Standard deviation of the activation noise. Default: 0
    #   n_neurons       Number of neurons in the network.
    #   return.global.act 
    #                   Sum total activation in each test-item (TRUE) or just 
    #                   the activation in the test-item (FALSE)
    #                   Default: FALSE
    
    test_act_sum <- data.frame (item = character(),
                                act = numeric ())
    
    for (ti in test_item_list){
        
        act <- abs(rnorm (n_neurons, 0, noise_sd_act))
        
        act_sum <- c()
        
        for (item in ti){
            
            current_input <- make_act_vector(item, n_neurons)
            act <- update_activation(act, res$w, current_input, 
                                     l_act, a, b, noise_sd_act,
                                     ...)
            
            if (return.global.act)
                act_sum <- c(act_sum, sum(act))
            else 
                act_sum <- c(act_sum, sum(act[ti]))
        }
        
        test_act_sum <- rbind (test_act_sum,
                               data.frame (item = paste (ti, collapse="-"),
                                           act = sum (act_sum)))
    }   
    
    test_act_sum <- test_act_sum %>%
        column_to_rownames ("item") %>% 
        t
    
    return (test_act_sum)
}

make_diff_score <- function (dat = ., 
                             col.name1,
                             col.name2,
                             normalize.scores = TRUE,
                             luce.rule = FALSE){
    
    if (luce.rule){
            d.score <- dat[,col.name1]
            normalize.scores <- TRUE
    } else {
        d.score <- dat[,col.name1] - dat[,col.name2]
    }
    
    if (any (d.score != 0) &&
        (normalize.scores))
        d.score = d.score / (dat[,col.name1] + dat[,col.name2])
    
    return (d.score)
    
}

summarize_condition <- function (dat,
                                 selected_cols,
                                 selected_cols_labels){ 
    
    sapply (selected_cols,
            function (X){
                c(M = mean (dat[,X]),
                  SE = mean (dat[,X]) / 
                      sqrt (length (dat[,X]) -1),
                  p.wilcox = wilcox.test (dat[,X])$p.value,
                  p.simulations = mean (dat[,X] > 0))
            },
            USE.NAMES = TRUE) %>% 
        #signif (3) %>%
        as.data.frame() %>%
        setNames (gsub ("\n", " ",
                        selected_cols_labels[selected_cols])) %>%
        # format_engr removes them otherwise
        rownames_to_column ("Statistic")
        #docxtools::format_engr(sigdig=3) 
    
}

format_p_simulations <- function (prop_sim){ 
    
    p_sim <- 100 * prop_sim
    
    min_diff_from_chance <- 
        get.min.number.of.correct.trials.by.binom.test(N_SIM)
    min_diff_from_chance <- 100 * min_diff_from_chance / N_SIM
    min_diff_from_chance <- min_diff_from_chance - 50
    
    p_sim <- ifelse (abs (p_sim-50) >= min_diff_from_chance,
                    paste ("({\\bf ", p_sim, " \\%})", 
                           sep =""), 
                    paste ("(", p_sim, " \\%)", 
                           sep ="") )
    
    return (p_sim)
}

get_sign_pattern_from_results <- function (l_act, dat){

    sign_pattern <- lapply (l_act, 
        function (CURRENT_L){
            tmp_p_values <- dat %>%
                filter (l_act == CURRENT_L) %>%
                dplyr::select (-c("l_act")) %>%
                column_to_rownames("Statistic")
            
            # Convert the proportion of simulations with a given outcome 
            # to a string; note that the proportion always gives the proportion 
            # for the majority pattern
            tmp_p_simulations <- tmp_p_values["p.simulations",] %>%
                mutate_all(format_p_simulations)
            
            # Extract the significance pattern into 
            # * + (significant preference for target)
            # * - (significant preference for foil)
            # * 0 (no significant preference)
            tmp_sign_pattern <- (tmp_p_values["p.wilcox",] <= .05) * 1
            tmp_sign_pattern <- tmp_sign_pattern * 
                sign(tmp_p_values["M",])
            
            tmp_sign_pattern <- tmp_sign_pattern %>%
                mutate_all(function (X) 
                    ifelse (X > 0, 
                            "+", 
                            ifelse (X < 0, 
                                    "-", 
                                    "0") ) )
            
            tmp_sign_pattern <- 
                tmp_sign_pattern %>%
                as.data.frame() %>%
                paste (., tmp_p_simulations, sep = " ") %>% 
                t () %>%
                as.data.frame() %>%
                setNames (names (tmp_sign_pattern)) %>% 
                add_column(l_act = CURRENT_L, .before = 1) %>%
                rownames_to_column("rowname") %>%
                dplyr::select (-c("rowname"))
            
            return (tmp_sign_pattern)
        }
    )
    
    sign_pattern <- do.call ("rbind", sign_pattern)
    
    sign_pattern
}

get_sign_pattern_for_plot %<a-% {
    # From https://github.com/kassambara/ggpubr/issues/79
    
    . %>%
        melt (id.vars = "l_act",
              variable.name= "ItemType",
              value.name = "d") %>%
        group_by (l_act, ItemType) %>%
        rstatix::wilcox_test(d ~ 1, mu = 0) %>%
        mutate (p.star = ifelse (p > .05, "",
                                 ifelse (p > .01,
                                         "*",
                                         ifelse (p > .001,
                                                 "**",
                                                 "***")))) %>%
        mutate(y.position = 0)
}
 
add_signif_to_plot <- function (gp, 
                                dat.df,
                                selected_cols){
 
    panel.info <- ggplot_build(gp)$layout$panel_params

    y.max <- lapply (panel.info, 
               function (X) max (X$y.range)) %>% 
        unlist %>%
        rep (., each = length (selected_cols))

    df.signif <- dat.df[,c("l_act",
                         selected_cols)] %>%
        get_sign_pattern_for_plot %>%
        mutate (y.position = y.max)
    
    gp <- gp + 
        ggpubr::stat_pvalue_manual(df.signif, 
                               label="p.star", 
                               xmin="l_act", 
                               xmax = "ItemType", 
                               remove.bracket = TRUE)

    return (gp)
}


format_theme %<a-% 
{
    theme_light() +
        theme(#text = element_text(size=20), 
            plot.title = element_text(size = 18, hjust = .5),
            axis.title = element_text(size=16),
            axis.text.x = element_text(size=14, angle = 45),
            axis.text.y = element_text(size=14),
            legend.title = element_text(size=16),
            legend.text = element_text(size=14))
}

remove_x_axis  %<a-% 
{
    
    theme(axis.title.x=element_blank(),
          axis.text.x=element_blank(),
          axis.ticks.x=element_blank())
}

rename_stuff_in_tables %<a-% {
    . %>%
        setNames (gsub ("l_act", "$\\\\lambda_a$", names (.))) %>%
        mutate (Statistic = compose (
            function (X) {gsub ("^M$", "*M*", X)},
            function (X) {gsub ("^SE$", "*SE*", X)},
            function (X) {gsub ("^p.wilcox$", "$p_{Wilcoxon}$", X)},
            function (X) {gsub ("^p.simulations$", "$P_{Simulations}$", X)}
        ) (Statistic)) 
}

find_chain_parts <- function() {
    # From https://stackoverflow.com/questions/42560389/get-name-of-dataframe-passed-through-pipe-in-r
    i <- 1
    while(!("chain_parts" %in% ls(envir=parent.frame(i))) && i < sys.nframe()) {
          i <- i+1
      }
    parent.frame(i)
}

print.plot <- function (p, 
                        p.name = NULL,
                        print.pdf = PRINT.INDIVIDUAL.PDFS){
    
    if (is.null (p.name)){
    # From https://stackoverflow.com/questions/42560389/get-name-of-dataframe-passed-through-pipe-in-r
    
     ee <- find_chain_parts()
     p.name <- deparse(ee$lhs)
    }
    
    if (print.pdf){
        
        pdf.name = sprintf ("figures/%s.pdf",
                            gsub ("\\.", "\\_",
                                  p.name))
        pdf (pdf.name)
        print (p)
        invisible(dev.off ())
    }
    
    print (p)
}

italisize_for_tex <- function (x = .){
    gsub("\\*(.+?)\\*", 
         "{\\\\em \\1}", 
         x, perl = TRUE)
}



## ----define-caption-functions--------------------------------------------------------------------------------------------------------------------------------------------

# Here we define functions to print the figure captions for consistency across figures

get.comparisons.for.caption <- function (experiment_type){
    
    if (experiment_type == "basic") {
        
        return ("Unit vs. Part-Unit: {\\em ABC} vs. {\\em BC:D} and {\\em ABC} vs. {\\em C:DE}; Rule-Unit vs. Class-Unit: {\\em AGC} vs. {\\em AGF} and {\\em AXC} vs. {\\em AXF}")
        
    } else if (experiment_type == "phantoms") {
        
        return ("Unit vs. Part-Unit: {\\em ABC} vs. {\\em BC:D} and {\\em ABC} vs. {\\em C:DE}; Phantom-Unit vs. Part-Unit: Phantom-Unit vs. {\\em BC:D} and Phantom-Unit vs. {\\em C:DE}; Unit vs. Phantom-Unit")
        
    } else {
        stop ("Unknown experiment type.")
    }
}

write.caption.diff.scores <- function (experiment_type, label, order, activation_type) {
    
    comparisons <- get.comparisons.for.caption(experiment_type)
    
    caption.text <- sprintf('\\label{fig:%s}Difference scores for items presented in {\\bf %s order}, different forgetting rates (0, 0.2, 0.4, 0.6, 0.8 and 1), and for the different comparisons (%s). The scores are calculated based the %s as a measure of the network\'s familiarity with the items. Significance is assessed based on Wilcoxon tests against the chance level of zero.',
                            label, order, comparisons, activation_type)
    
    return (caption.text)
}

write.caption.p.sim <- function (experiment_type, label, order, activation_type) {
    
    comparisons <- get.comparisons.for.caption(experiment_type)
    
    caption.text <- sprintf('\\label{fig:%s}Percentage of simulations with a preference for the target items for items presented in {\\bf %s order}, different forgetting rates (0, 0.2, 0.4, 0.6, 0.8 and 1) and for the different comparisons (%s). The simulations are assessed based on the %s. The dashed line shows the minimum percentage of simulations that is significant based on a binomial test.',
                            label, order, comparisons, activation_type)
    
    return (caption.text)
}



## ----define-spectral-analysis-functions----------------------------------------------------------------------------------------------------------------------------------

make.reference.waves <- function (T = ., length.out){
    
    # Base reference wave of a cosine with phase 0
    reference.wave <- cos (2 * pi * (0:(T-1)) / T)
    
    # Now make the matrix of reference waves with different phases
    m.reference.waves <- matrix (nrow = T, ncol = T)
    for (phase in 0:(T-1)){
        m.reference.waves[,phase+1] <- cyclic.shift (reference.wave, -phase)
    }
    
    m.reference.waves <- map (1:(length.out/T),
                              ~ m.reference.waves) %>% 
        do.call (rbind, .)
    
    ts (m.reference.waves, names = paste0 ("cos.phase", 0:(T-1)))
    
    
}

extract.relevant.phase <- function (phase = ., row, reference.ts = 1, comparison.ts = NULL, phase.units = c("parts.of.cycle", "radians", "degrees")) {
    
    phase.units <- match.arg (phase.units)
    
    # According to the spectrum documentation 
    # Column i + (j - 1) * (j - 2)/2 of phase contains the squared coherency between columns i and j of x, where i < j.
    
    if (is.null (comparison.ts)) {
        # Inverting the formula above for i = n.time.series - 1 and j = n.time.series
        n.time.series <- 0.5 * (1 + sqrt (8 * ncol (phase) + 1))
        
        comparison.ts <- 2:n.time.series 
    }
    
    relevant.phase <- phase[row,reference.ts + (comparison.ts - 1) * (comparison.ts - 2)/2] 
    
    case_when(
        phase.units == "parts.of.cycle" ~ relevant.phase / (2 * pi),
        phase.units == "degrees" ~ relevant.phase / (2 * pi) * 360, 
        phase.units == "radians" ~ relevant.phase
    )
    
    
    
}


get.act.freq.phase <- function (x = ., T.reference.wave = 3, phase.units = c("parts.of.cycle", "radians", "degrees")){
    # Arguments
    # x: Time series
    # T.reference.wave: Period of the reference time series
    # phase.units: Units for Phase 
    
    
    # average activation 
    x %>% 
        matrix (ncol = 3, byrow = TRUE) %>% 
        colMeans() -> act.in.words
        
    
    # Spectral analysis
    phase.units <- match.arg (phase.units)
    
    observed.wave <- as.ts (x)
    
    # Default is cosine with maximum on third item
    reference.waves <- make.reference.waves(T.reference.wave, length (x))

    spec.observed.reference <- spectrum (ts.intersect(observed.wave,
                                                      reference.waves),
                                         plot = FALSE)

    max.index <- apply (spec.observed.reference$spec, 2, which.max)
    
    max.freq <- spec.observed.reference$freq[max.index[1]]

    phase <- extract.relevant.phase (spec.observed.reference$phase,
                                     max.index[1],
                                     phase.units = phase.units)
                                     

    c (max.freq, phase, act.in.words) %>% 
        matrix (nrow = 1) %>% 
        as.data.frame %>% 
        setNames(c("freq", paste0 ("phase.", colnames (reference.waves)), paste0("act.s", 1:T.reference.wave)))
    
}




## ----basic-experiment-run, echo = FALSE----------------------------------------------------------------------------------------------------------------------------------

fam_basic <- matrix (rep(1:(N_WORDS * N_SYLL_PER_WORD), 
                         N_REP_PER_WORD), 
                     byrow = TRUE, ncol=N_SYLL_PER_WORD)

# 3 syllable test items
test_items_basic <- list (1:3,        # W
                          2:4,        # PW (BCA) 
                          3:5        # PW (CAB)
                          # No longer needed
                          # c(1,4,3),   # RW (moved middle syllable)
                          # c(1,4,9),   # CW (moved middle syllable)
                          # c(1,19,3),  # RW (new middle syllable)
                          # c(1,19,9)   # CW (new middle syllable)
)

# test_items_basic <- c(test_items_basic, 
#                       lapply (test_items_basic, 
#                               rev))

# 2 syllable test items as in Flo et al. 
test_items_basic_2syll <- list (1:2,   # W
                               2:3)   # PW (BCA)

# We test-items in two ways: by recording the activation in the test-items themselves
# and by recording the activation in the entire network (_global)
# We report only the global activation

# For 3 syllable test items
test_act_sum_basic_list <- list ()
test_act_sum_basic_global_list <- list ()

# For 2 syllable test items
test_act_sum_basic_2syll_list <- list ()
test_act_sum_basic_2syll_global_list <- list ()


spectral_ana_basic_list <- list ()



for (current_l in L_ACT){
    # Sample through forgetting values 
    
    current_test_act_sum_basic <- data.frame()
    current_test_act_sum_basic_global <- data.frame()

    current_test_act_sum_basic_2syll <- data.frame()
    current_test_act_sum_basic_2syll_global <- data.frame()
    
    current_spectral_ana_basic <- data.frame ()
    
    for (i in 1:N_SIM){
        
        res <- familiarize (stream = fam_basic,
                            l_act = current_l, a = A, b = B, noise_sd_act = NOISE_SD_ACT,
                            r = R, l_w = L_W, noise_sd_w = NOISE_SD_W,
                            n_neurons = 19)
        
        
        #plot (res$act_sum, type="l")
        
        # Record results from spectral analysis 
        current_spectral_ana_basic <- rbind (current_spectral_ana_basic,
                                         get.act.freq.phase (res$act_sum %>% 
                                                             # Remove the first 10 presentations of each word
                                                             tail (3 * N_WORDS * (N_REP_PER_WORD - N_REP_PER_WORD_BURNIN)), 
                                                         phase.units = "degrees") %>% 
                                             mutate (l_act = current_l,
                                                     a = A, b = B, 
                                                     noise_sd_act = NOISE_SD_ACT,
                                                     n_neurons = 19,
                                                     .before = 1))
                                                     
                                            
                                         
        
        # Record activation in test-items for 3 syllable items 
        current_test_act_sum_basic <- rbind (current_test_act_sum_basic,
                                             test_list (test_item_list = test_items_basic,
                                                        w = res$w,
                                                        l_act = current_l, a = A, b = B, 
                                                        noise_sd_act = NOISE_SD_ACT,
                                                        n_neurons = 19,
                                                        return.global.act = FALSE)) 
        
        # Record global activation in network for 3 syllable items 
        current_test_act_sum_basic_global <- rbind (current_test_act_sum_basic_global,
                                             test_list (test_item_list = test_items_basic,
                                                        w = res$w,
                                                        l_act = current_l, a = A, b = B, 
                                                        noise_sd_act = NOISE_SD_ACT,
                                                        n_neurons = 19,
                                                        return.global.act = TRUE)) 
        
        
        # Record activation in test-items for 2 syllable items 
        current_test_act_sum_basic_2syll <- rbind (current_test_act_sum_basic_2syll,
                                             test_list (test_item_list = test_items_basic_2syll,
                                                        w = res$w,
                                                        l_act = current_l, a = A, b = B, 
                                                        noise_sd_act = NOISE_SD_ACT,
                                                        n_neurons = 19,
                                                        return.global.act = FALSE)) 
        
        # Record global activation in network for 2 syllable items 
        current_test_act_sum_basic_2syll_global <- rbind (current_test_act_sum_basic_2syll_global,
                                             test_list (test_item_list = test_items_basic_2syll,
                                                        w = res$w,
                                                        l_act = current_l, a = A, b = B, 
                                                        noise_sd_act = NOISE_SD_ACT,
                                                        n_neurons = 19,
                                                        return.global.act = TRUE)) 
        
    }
    
    # End of forgetting sampling loop
    
    spectral_ana_basic_list[[1 + length (spectral_ana_basic_list)]] <- 
        current_spectral_ana_basic
    
    test_act_sum_basic_list[[1 + length (test_act_sum_basic_list)]]  <- 
        current_test_act_sum_basic
    
    test_act_sum_basic_global_list[[1 + length (test_act_sum_basic_global_list)]]  <- 
        current_test_act_sum_basic_global
    
    test_act_sum_basic_2syll_list[[1 + length (test_act_sum_basic_2syll_list)]]  <- 
        current_test_act_sum_basic_2syll
    
    test_act_sum_basic_2syll_global_list[[1 + length (test_act_sum_basic_2syll_global_list)]]  <- 
        current_test_act_sum_basic_2syll_global
}

# Combine results from different forgetting rates
# Spectral analysis
spectral_ana_basic <- 
    do.call (rbind,
             spectral_ana_basic_list)

# Test lists with 3 syllable words
test_act_sum_basic <- 
    do.call (rbind, 
             test_act_sum_basic_list)

test_act_sum_basic <- test_act_sum_basic %>% 
    add_column(l_act = rep (L_ACT,
                            sapply (test_act_sum_basic_list,
                                    nrow)),
               .before = 1
    )

test_act_sum_basic_global <- 
    do.call (rbind, 
             test_act_sum_basic_global_list)

test_act_sum_basic_global <- test_act_sum_basic_global %>% 
    add_column(l_act = rep (L_ACT,
                            sapply (test_act_sum_basic_global_list,
                                    nrow)),
               .before = 1
    )

# Test lists with 2 syllable words
test_act_sum_basic_2syll <- 
    do.call (rbind, 
             test_act_sum_basic_2syll_list)

test_act_sum_basic_2syll <- test_act_sum_basic_2syll %>% 
    add_column(l_act = rep (L_ACT,
                            sapply (test_act_sum_basic_2syll_list,
                                    nrow)),
               .before = 1
    )

test_act_sum_basic_2syll_global <- 
    do.call (rbind, 
             test_act_sum_basic_2syll_global_list)

test_act_sum_basic_2syll_global <- test_act_sum_basic_2syll_global %>% 
    add_column(l_act = rep (L_ACT,
                            sapply (test_act_sum_basic_2syll_global_list,
                                    nrow)),
               .before = 1
    )


## ----basic-experiment-global-create_diff---------------------------------------------------------------------------------------------------------------------------------

diff_basic_global <- cbind(
    l_act = data.frame (l_act = test_act_sum_basic_global$l_act),
    
    # Adjacent FW TP: Words vs. Part-Words (Forward)
    w_pw1_fw = make_diff_score(test_act_sum_basic_global,
                               "1-2-3", "2-3-4",
                               TRUE),
    w_pw2_fw = make_diff_score(test_act_sum_basic_global,
                               "1-2-3", "3-4-5",
                               TRUE)
    
    # Unused 
    # # Adjacent BW TP: Words vs. Part-Words (Backward)
    # w_pw1_bw = make_diff_score(test_act_sum_basic_global,
    #                            "3-2-1", "4-3-2",
    #                            TRUE),
    # w_pw2_bw = make_diff_score(test_act_sum_basic_global,
    #                            "3-2-1", "5-4-3",
    #                            TRUE),
    # 
    # # Non-adjacent FW TP: Rule-Words vs. Class-Words (Forward)
    # rw_cw_fw1 = make_diff_score(test_act_sum_basic_global,
    #                             "1-4-3", "1-4-9",
    #                             TRUE),
    # rw_cw_fw2 = make_diff_score(test_act_sum_basic_global,
    #                             "1-19-3", "1-19-9",
    #                             TRUE),
    # 
    # # Non-adjacent BW TP: Rule-Words vs. Class-Words (Backward)
    # rw_cw_bw1 = make_diff_score(test_act_sum_basic_global,
    #                             "3-4-1", "9-4-1",
    #                             TRUE),
    # rw_cw_bw2 = make_diff_score(test_act_sum_basic_global,
    #                             "3-19-1", "9-19-1",
    #                             TRUE)
) %>%
    as.data.frame()


#boxplot (diff_basic, ylim=c(0, .2))


## ----basic-experiment-global-create-plot_diff-fw-------------------------------------------------------------------------------------------------------------------------
selected_cols_fw <- c(
    "w_pw1_fw", "w_pw2_fw")
#    "rw_cw_fw1", "rw_cw_fw2")

selected_cols_labels <- c(
    w_pw1_fw = "ABC vs\nBC:D",
    w_pw2_fw = "ABC vs\nC:DE"
    # rw_cw_fw1 = "AGC vs\nAGF", 
    # rw_cw_fw2 = "AXC vs\nAXF",
    
    # w_pw1_bw = "ABC vs\nBC:D",
    # w_pw2_bw = "ABC vs\nC:DE",
    # rw_cw_bw1 = "AGC vs\nAGF", 
    # rw_cw_bw2 = "AXC vs\nAXF"
)

diff_basic_global_fw_plot <- 
    diff_basic_global[,c("l_act",
                  selected_cols_fw)] %>%
    melt (id.vars = "l_act",
          variable.name= "ItemType",
          value.name = "d") %>%
    ggplot(aes(x=ItemType, y=d, fill=ItemType))+
    format_theme + 
    remove_x_axis + 
    labs (#title = "Forward TPs",
          y = TeX("\\frac{Type_1 - Type_2}{Type_1 + Type_2}")) +
#     scale_x_discrete(name = "Item Type",
#                      breaks = 1:4,                 
#                      labels=                         selected_cols_labels[selected_cols_fw]) + 
    facet_wrap(~l_act, scales = "free_y") +
    scale_fill_discrete(name = element_blank(), 
                        labels = selected_cols_labels[selected_cols_fw]) + 
    theme(legend.position = "bottom",
          legend.direction = "horizontal") + 
    geom_boxplot()  

diff_basic_global_fw_plot <- add_signif_to_plot(
    diff_basic_global_fw_plot,
    diff_basic_global,
    selected_cols_fw)




## ----basic-experiment-global-evaluate_diff-fw----------------------------------------------------------------------------------------------------------------------------

diff_basic_global_fw_p_values <- 
    lapply (L_ACT,
            function (CURRENT_L){
                diff_basic_global %>%
                    filter (l_act == CURRENT_L) %>% 
                    summarize_condition(., 
                                        selected_cols_fw,
                                        selected_cols_labels) %>% 
                    add_column(l_act = CURRENT_L, .before = 1)
            }
    ) 

diff_basic_global_fw_p_values <- do.call ("rbind", 
                                   diff_basic_global_fw_p_values )



## ----basic-experiment-global-create-plot_p_sim-fw------------------------------------------------------------------------------------------------------------------------

p_sim_basic_global_fw_plot <- diff_basic_global_fw_p_values %>%
    filter (Statistic == "p.simulations") %>%
    dplyr::select(-c("Statistic")) %>%
        melt (id.vars = "l_act",
          variable.name= "ItemType",
          value.name = "P")  %>%
    ggplot(aes(x=ItemType, y= 100 * P, fill = ItemType))+
    format_theme + 
    remove_x_axis + 
    labs (#title = "Forward TPs",
          y = "Percentage of Simulations") +
#    scale_x_discrete(name = "Item Type",
#                     breaks = 1:4, 
                     #labels=selected_cols_labels[selected_cols_bw]) + 
    facet_wrap(~l_act, scales = "fixed") + 
    scale_fill_discrete(name = element_blank(), 
                        labels = unname (selected_cols_labels[selected_cols_fw])) + 
    theme(legend.position = "bottom",
          legend.direction = "horizontal") + 
    geom_bar(stat = "identity") + 
    geom_abline(intercept = 
                    get.min.number.of.correct.trials.by.binom.test(N_SIM)/N_SIM * 100, 
                slope = 0,
                linetype = "dashed") + 
    geom_text (aes (label=100*P, y=5))




## ----basic-experiment-global-create-plot_combined-fw-plot, include = TRUE, fig.cap = "(a) Difference scores between words and part-words for different forgetting rates (0, 0.2, 0.4, 0.6, 0.8 and 1). The scores are calculated based the global activation as a measure of the network's familiarity with the items. Significance is assessed based on Wilcoxon tests against the chance level of zero. (b). Percentage of simulations with a preference for words for different forgetting rates (0, 0.2, 0.4, 0.6, 0.8 and 1). The simulations are assessed based on the global activation in the network. The dashed line shows the minimum percentage of simulations that is significant based on a binomial test.", fig.height=8----

# plot_grid_with_title(diff_basic_global_fw_plot +
#               theme (title = element_blank()),
#           p_sim_basic_global_fw_plot +
#               theme (title = element_blank()),
#           nrow=2,
#           labels = "auto",
#           title.label = "Forward TPs") %>%
#     print.plot (p.name = "basic_global_combined_fw")



ggpubr::ggarrange (diff_basic_global_fw_plot + 
                       theme (title = element_blank()),
                   p_sim_basic_global_fw_plot +
                       theme (title = element_blank()), 
                   nrow=2,
                   labels = "auto",
                   common.legend = TRUE,
                   legend = "bottom") %>%
     print.plot (p.name = "basic_global_combined_fw")



## ----basic-experiment-global-evaluate_diff-print, results='hide'---------------------------------------------------------------------------------------------------------
diff_basic_global_fw_p_values %>%
    rename_stuff_in_tables %>%
    mutate(Statistic = italisize_for_tex (Statistic)) %>%    
    mutate(Statistic = italisize_for_tex (Statistic)) %>%
    docxtools::format_engr(sigdig=3) %>%
    knitr::kable(
        "latex", 
        longtable = TRUE,
        booktabs = TRUE, 
        escape = FALSE,
        caption = '\\label{tab:basic_diff} Detailed results for the different forgetting rates for the word vs. part-word comparison (*ABC* vs. *BC:D* and *ABC* vs. *C:DE*), and using the global activation as a measure of the network\'s familiarity with the items. $p_{Wilcoxon}$ represents the *p* value of a Wilcoxon test on the difference scores against the chance level of zero. $P_{Simulations}$ represents the proportion of simulations showing positive difference scores.') %>%
    kableExtra::kable_styling(latex_options = c("striped", 
#                                                "scale_down",
                                                "repeat_header")) %>% 
    kableExtra::kable_classic_2()


## ----basic-experiment-global-sign-pattern-print, results='hide'----------------------------------------------------------------------------------------------------------
get_sign_pattern_from_results(L_ACT, 
                              diff_basic_global_fw_p_values) %>%
    setNames(., gsub("l_act", "$\\\\lambda_a$", names(.))) %>%
    knitr::kable(
        "latex", 
        longtable = FALSE,
        booktabs = TRUE, 
        escape = FALSE,
        caption = '\\label{tab:basic_global_sign_pattern} Pattern of significance for the different forgetting rates and comparisons (Unit vs. Part-Unit: *ABC* vs. *BC:D* and *ABC* vs. *C:DE*; Rule-Unit vs. Class-Unit: *AGC* vs. *AGF* and *AXC* vs. *AXF*), for items presented in forward and backward order, and using the global activation as a measure of the network\'s familiarity with the items. +, - and 0 represent, respectively, a significant preference for the target item, a significant preference against the target item, or no significant preference, as evaluated by Wilcoxon tests on the relevant difference scores. Numbers indicate the proportion of simulations preferring target-items; bold-face numbers indicate significance in a binomial test.') %>%
    kableExtra::kable_styling(latex_options = c("striped", 
                                                "scale_down",
                                                "repeat_header")) %>%
    kableExtra::kable_classic_2()



## ----basic-experiment-global-print-act-in-words-plot, fig.cap = "Average total network activation for different syllables in 100 simulations in Endress and Johnson's network during the familiarization with a stream following Saffran et al. (1996). The facets show different forgetting rates. The results reflect the network behavior after the first 20 presentations of each word. "----

spectral_ana_basic %>% 
    filter (l_act > .2) %>%
    filter (l_act < 1) %>% 
    pivot_longer(starts_with("act.s"),
                 names_to = "syllable",
                 values_to = "act") %>% 
    mutate (syllable = str_remove(syllable, "act.") %>% 
                factor ) %>% 
    mutate (l_act = factor (l_act)) %>% 
    ggplot (aes (x=syllable, y = act)) + 
    theme_linedraw(14) + 
    geom_violin (alpha = .5,
                 fill = "#5588CC",
                 col="#5588CC") +
    scale_x_discrete("Syllable",
                     labels = ~ str_wrap(.x, 15),
                     guide = guide_axis(angle = 60)) +
    scale_y_continuous("Average activation") +# , limits = 0:1) + 
    stat_summary(fun.data=mean_se, 
                 geom="pointrange", color="#cc556f")  +
    facet_wrap(l_act ~ ., scales = "free_y")



## ----basic-experiment-global-create-freq-plot----------------------------------------------------------------------------------------------------------------------------
spectral_ana_basic %>% 
    mutate (l_act = factor (l_act)) %>% 
    ggplot (aes (x=l_act, y = freq)) + 
    theme_linedraw(14) + 
    geom_violin (alpha = .5,
                 fill = "#5588CC",
                 col="#5588CC") +
    scale_x_discrete(TeX ("Forgetting ($\\Lambda$)"),
                     labels = ~ str_wrap(.x, 15),
                     guide = guide_axis(angle = 60)) +
    scale_y_continuous("Frequency") +# , limits = 0:1) + 
    stat_summary(fun.data=mean_se, 
                 geom="pointrange", color="#cc556f") -> 
    plot.basic.experiment.freq


## ----basic-experiment-global-create-phase-plot---------------------------------------------------------------------------------------------------------------------------
spectral_ana_basic %>% 
    pivot_longer(starts_with("phase"),
                 names_to = "reference.phase",
                 values_to = "relative.phase") %>% 
    mutate (reference.phase = str_remove(reference.phase, "phase.cos.phase") %>% 
                factor ) %>% 
    mutate (reference.phase = plyr::revalue (reference.phase, 
                                       c("0" = "Word-initial cosine",
                                       "1" = "Word-medial cosine",
                                       "2" = "Word-final cosine"))) %>% 
    mutate (l_act = factor (l_act)) %>% 
    ggplot (aes (x=l_act, y = relative.phase)) + 
    theme_linedraw(14) + 
    geom_violin (alpha = .5,
                 fill = "#5588CC",
                 col="#5588CC") +
    scale_x_discrete(TeX ("Forgetting ($\\Lambda$)"),
                     labels = ~ str_wrap(.x, 15),
                     guide = guide_axis(angle = 60)) +
    scale_y_continuous("Relative phase (degrees)") +# , limits = 0:1) + 
    stat_summary(fun.data=mean_se, 
                 geom="pointrange", color="#cc556f")  +
    facet_grid(reference.phase ~ ., labeller = labeller (reference.phase = ~ str_wrap(.x, 12))) -> 
    plot.basic.experiment.phase


## ----basic-experiment-global-print-freq-phase-plot, fig.cap = "Spectral analysis of the total network activation in 100 simulations in Endress and Johnson's network during the familiarization with a stream following Saffran et al. (1996). The results reflect the network behavior after the first 20 presentations of each word. (a) Maximal frequency as a function of the forgeting rates. For forgetting rates where learning takes place, the dominant frequency is 1/3, and thus corresponds to the word length. (b) Relative phase (in degrees) at the maximal frequency of the total network activation relative to a consine function with its maximum at word-intial syllables (top), word-second syllables (middle) and word-final syllables (bottom). For forgetting rates where learning takes place, the total activation is in phase with a cosine with its maximum on the word-final syllable."----

grid.arrange.tag(plot.basic.experiment.freq,
                 plot.basic.experiment.phase,
                 ncol = 2)



## ----basic-experiment-global-print-difference-between-parts-of-word------------------------------------------------------------------------------------------------------
# We recorded the activation when the network was exposed to the first two syllables of a word
# and when it was exposed to the last two syllables of a word


test_act_sum_basic_2syll_global %>% 
    mutate (d = `1-2` - `2-3`) %>% 
    group_by(l_act) %>% 
    summarize (d.m = mean (d), d.se = se (d), p = wilcox.p (d)) %>% 
    kable (caption = "Activation difference between items composed of the first two items of a word and the last two items of a word, when these bigrams were presented in isolation.",
           booktabs = TRUE) %>% 
    kableExtra::kable_classic_2()
    

    



## ----basic-experiment-global-print-act-after-2syll-plot, fig.cap = "Average difference in the total network activation for the first two syllables of a word (AB) and the first to syllables of a part-word (BC)  in 100 simulations in Endress and Johnson's network after  familiarization with a stream following Saffran et al. (1996). The results reflect the network behavior after the first 20 presentations of each word. "----

bind_rows (test_act_sum_basic_2syll %>% 
               mutate (d = `1-2` - `2-3`) %>% 
               mutate (act.type = "Activation in test items"),
    test_act_sum_basic_2syll_global %>% 
               mutate (d = `1-2` - `2-3`) %>% 
               mutate (act.type = "Global activation")
) %>% 
    filter (l_act > 0.2) %>% 
    mutate (l_act = factor (l_act)) %>% 
    ggplot (aes (x = l_act, y = d)) + 
    theme_linedraw(14) + 
    geom_violin (alpha = .5,
                 fill = "#5588CC",
                 col="#5588CC") +
    scale_x_discrete(TeX ("Forgetting ($\\Lambda$)"),
                     labels = ~ str_wrap(.x, 15),
                     guide = guide_axis(angle = 60)) +
    scale_y_continuous("Activation difference AB - BC") +# , limits = 0:1) #+ 
    stat_summary(fun.data=mean_se, 
                 geom="pointrange", color="#cc556f") +
    facet_wrap(act.type ~ ., scales = "fixed") #+
    #ggpubr::stat_pvalue_manual()



## ----list-parameters2, ref.label='list-parameters', echo=FALSE, results='markup', caption='\\label{tab:params}: Simulation parameters.'----------------------------------


## ----basic-experiment-global-evaluate_diff-print2, ref.label='basic-experiment-global-evaluate_diff-print', echo=FALSE, results='markup'---------------------------------



## ----save-stuff, echo = FALSE--------------------------------------------------------------------------------------------------------------------------------------------
#save.image("/Users/endress/Experiments/TP_model/tp_model.RData")

