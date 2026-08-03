## THIS IS THE FILE THAT I'M UPDATING FROM ZONGBO'S WORK IN "main_optimization_updated".

###############################################################################################
###################### PROFOUND Naloxone Distribution model #### 2024 #########################
# Updated Main module

rm(list = ls())
library(argparser)
library(dplyr)
library(tictoc)
library(openxlsx)
library(abind)
library(tictoc)
library(readr) #Added to be able to read the Naloxone distributions and append to OEND matrix
library(readxl) # To read the Excel file of the Calibrated Results
source("transition_probability.R")
source("microsim_seedfixed.R")
source("decision_tree.R")
source("data_input.R")
source("naloxone_availability.R")
source("cost_effectiveness.R")

# Keep track of the job_id: 
args <- commandArgs(trailingOnly = TRUE)  # Read command-line arguments
print(args)
job_id <- as.numeric(args[2])  # Job ID (e.g., 1 or 2)
print(job_id)


yr_start <- 2016 # starting year of simulation
yr_end <- 2026 # end year of simulation (also the year for evaluation)
discount.rate <- 0.03 # discounting of costs by 3%

naloxone_data <- read.csv("naloxone_kit_distributions.csv")
# Add number to iteration if you want to start in middle of naloxone_kit_distributions.csv
iteration <- floor((job_id - 1) / 100) + 1

# Only remove rows if iteration is greater than 1
if (iteration > 1) {
  rows_to_remove = 13312 * (iteration - 1)
  naloxone_data <- naloxone_data[-(1:rows_to_remove), ]
}

# Add three rows to the NxDataPharm matrix: 

# Create the new rows as a data frame
# ADD THE 2023 values
new_rows <- data.frame(year = c(2024, 2025, 2026), pe = c(19719, 19719, 19719))

# Combine the new rows with the existing matrix
params$NxDataPharm <- rbind(params$NxDataPharm, new_rows)

# BELOW code seems to generate error, but removing it leads to issues later in the file. Need to navigate around 
# this problem while maintaining this code block. 

args <- arg_parser("arguments")
args <- add_argument(args, "--seed", help = "seed for random numbers", default = 2021)
args <- add_argument(args, "--regional", help = "flag to run regional model", flag = TRUE)
args <- add_argument(args, "--outfile", help = "file to store outputs", default = "OverdoseDeath_RIV1_0.csv")
args <- add_argument(args, "--ppl", help = "file with initial ppl info", default = "Inputs/init_pop.rds")
# Adding new line to include an argument for getting the correct job_id for slurm array:
args <- add_argument(args, "--args", help = "file getting job_ids for slurm array", default = 1)
argv <- parse_args(args)
seed <- as.integer(argv$seed)
init_ppl.file <- argv$ppl

source("io_setup.R")


##################################### Run simulation ######################################################
############## RI modeling analysis: Teva Opioid Settlement and Solitary Drug Use Interventions ##################
###########################################################################################################
# Define different program scenarios for distributing additional 10,000 kits and initialize matrices and arrrays for results
# Define the number of cores to use (adjust as needed)
num_cores <- 32  # Assuming n.reds = 7
n.reds <- 416*num_cores    # number of redistribution schemes
scenario.name <- c("Status Quo", paste("Redistribution_", c(1:n.reds), sep = ""))  #name strategies for different distribution 


# DELETED "combine_custom" FUNCTION BECAUSE IT WASN"T USED ANYWHERE IN PROGRAM

yr_start <- 2016 # simulation first year
yr_end <- 2026 # simulation last year
yr_int <- 2024
agent_states <- c("preb", "il.lr", "il.hr", "inact", "NODU", "relap", "dead") # vector for state names
v.oustate <- c("preb", "il.lr", "il.hr") # vector for active opioid use state names
num_states <- length(agent_states) # number of states
num_years <- yr_end - yr_start + 1
timesteps <- 12 * num_years # number of time cycles (in month)
num_regions <- length(v.region) # number of regions
discount.rate <- 0.03 # discounting of costs by 3%

# Deleting "number_of_parameters" because it doesn't seem to be needed. 

# OUTPUT for analysis
mat.od.death <- matrix(nrow = n.reds, ncol = yr_end - yr_int + 1 + 39) # 39 for number of cities/towns

# OUTPUT matrices and vectors
v.od <- rep(0, times = timesteps) # count of overdose events at each time step
v.oddeath <- rep(0, times = timesteps) # count of overdose deaths at each time step
v.oddeath.w <- rep(0, times = timesteps) # count of overdose deaths that were witnessed at each time step
m.oddeath <- matrix(0, nrow = timesteps, ncol = num_regions)
colnames(m.oddeath) <- v.region
v.odpriv <- rep(0, times = timesteps) # count of overdose events occurred at private setting at each time step
v.odpubl <- rep(0, times = timesteps) # count of overdose events occurred at public setting at each time step
v.deathpriv <- rep(0, times = timesteps) # count of overdose deaths occurred at private setting at each time step
v.deathpubl <- rep(0, times = timesteps) # count of overdose deaths occurred at public setting at each time step
v.nlxused <- rep(0, times = timesteps) # count of naloxone kits used at each time step
v.str <- c("SQ", "expand", "program") # store the strategy names
cost.item <- c("TotalCost", "NxCost")
cost.matrix <- matrix(0, nrow = timesteps, ncol = length(cost.item))
colnames(cost.matrix) <- cost.item
m.oddeath.fx <- rep(0, times = timesteps) # count of overdose deaths with fentanyl present at each time step
m.oddeath.op <- rep(0, times = timesteps) # count of overdose deaths among opioid users at each time step
m.oddeath.st <- rep(0, times = timesteps) # count of overdose deaths among stimulant users at each time step
m.EDvisits <- rep(0, times = timesteps) # count of opioid overdose-related ED visits at each time step
m.oddeath.hr <- rep(0, times = timesteps) # count of overdose deaths among high-risk opioid users (inject heroin) at each time step
m.oddeath.preb <- m.oddeath.il.lr <- m.oddeath.il.hr <- m.nlx.mn.OEND <- m.nlx.mn.Pharm <- rep(0, times = timesteps)   # count of overdose deaths stratified by risk group each time step

## Initialize the study population - people who are at risk of opioid overdose
ppl_info <- c("sex", "race", "age", "residence", "curr.state", "OU.state", "init.age", "init.state", "ever.od", "fx")
init_ppl <- readRDS(paste0("Inputs/init_pop.rds"))

# array.od.death <- matrix(0, nrow = (yr_end - yr_int + 1), ncol = length(scenario.name))
# array.od.death.wtns <- array.od.death
# array.F2Aratio <- matrix(0, nrow = num_years, ncol = length(scenario.name))
# array.od.death.rgn.26 <- matrix(0, nrow = num_regions, ncol = length(scenario.name))
# array.od.death.rgn.23 <- array.od.death.rgn.25 <- array.od.death.rgn.30
# array.oend.nlx.rgn <- array.od.death.rgn.36

# READ IN DATA BEFORE FOR LOOP

years_to_keep <- as.character(2015:2023)

row_counter <- 1

# ADD TIME TO DETERMINE LENGTH OF RUN

#start_time <- Sys.time()

# PARALLEL STUFF: 

# Load parallel package
library(parallel)


# Define the task to parallelize
process_scenario <- function(i) {
  # Initialize the matrix for results of this core
  results <- matrix(0, nrow = 1, ncol = 42)  # Assuming 42 columns in mat.od.death
  
  # Process the ith scenario
  params$NxOEND.matrix <- params$NxOEND.matrix[rownames(params$NxOEND.matrix) %in% years_to_keep, ]
  naloxone_distribution <- as.numeric(naloxone_data[i, -1])  # Remove the first column (year or index) and convert to double
  naloxone_distribution <- matrix(naloxone_distribution, nrow = 1)
  
  # Add "naloxone_distribution" this to end of output
  new_rows <- data.frame(matrix(rep(naloxone_distribution, 3), nrow = 3, byrow = TRUE))
  rownames(new_rows) <- c("2024", "2025", "2026")  # Set years as row names
  colnames(new_rows) <- colnames(naloxone_data)[-1]  # Set column names to match existing data (excluding year column)
  params$NxOEND.matrix <- rbind(params$NxOEND.matrix, new_rows)
  
  remainder <- job_id %% 100
  result <- ifelse(remainder == 0, 100, remainder)
  vparameters.temp <- sim.data.ls[[result]]  # Assuming you want the same parameters for all iterations
  vparameters.temp$covid.2inact <- 0
  vparameters.temp$covid.oud.fx <- 1
  vparameters.temp$covid.NOUD.fx <- 1.23
  vparameters.temp$mor_nx <- 0
  vparameters.temp$NxOEND.matrix <- params$NxOEND.matrix
  vparameters.temp$NxDataPharm <- params$NxDataPharm
  
  sim_sq <- MicroSim(
    init_ppl, params = vparameters.temp, timesteps, agent_states,
    discount.rate, PT.out = FALSE, strategy = "SQ", seed = sim.seed[[result]]
  )
  
  # Calculate the values to go in the first 3 columns
  results[1, 1:3] <- colSums(matrix(rowSums(sim_sq$m.oddeath[((yr_int - yr_start) * 12 + 1):timesteps, ]), nrow = 12))
  
  # Add naloxone_distribution values to the remaining 39 columns
  results[1, 4:42] <- naloxone_distribution
  
  # Return the results for this iteration
  return(results)
}

# Create a cluster of cores
cl <- makeCluster(num_cores)

# Load necessary libraries on each worker node
clusterEvalQ(cl, {
  library(dplyr) # Ensure %>% is available
  library(magrittr) # If %>% is not from dplyr
})

# Export variables and functions to the cluster
clusterExport(cl, varlist = c(
  "params", "num_years", "years_to_keep", "naloxone_data", "sim.data.ls",
  "init_ppl", "MicroSim", "timesteps", "agent_states", "discount.rate",
  "yr_start", "yr_int", "sim.seed", "trans.prob", "num_states", "samplev", "v.oustate",
  "v.od", "crosstab", "nlx.avail.algm", "decision_tree", "sample.dic", "v.oddeath", "v.oddeath.w",
  "v.odpriv", "v.odpubl", "v.deathpriv", "v.deathpubl", "v.nlxused", "m.oddeath.fx", "m.oddeath.op", 
  "m.oddeath.st", "m.EDvisits", "m.oddeath.preb", "m.oddeath.il.lr", "m.oddeath.il.hr", "m.nlx.mn.OEND",
  "m.nlx.mn.Pharm", "m.oddeath", "m.odnonfatal", "Costs", "cost.matrix", "m.oddeath.hr", "job_id"
))

# Run the tasks in parallel using parLapply
results_list <- parLapply(cl, 1:n.reds, process_scenario)

# Stop the cluster
stopCluster(cl)

# Combine results into the mat.od.death matrix
mat.od.death <- do.call(rbind, results_list)

# Define names for the first 3 columns
first_col_names <- paste0("od_death_", yr_int:(yr_int + 2))  # Assuming 3 years in sequence

# Define names for the remaining 39 columns from naloxone_distribution
naloxone_col_names <- colnames(read.csv("naloxone_kit_distributions.csv", nrows = 1))[-1]

# Combine into a single vector of column names
colnames(mat.od.death) <- c(first_col_names, naloxone_col_names)

# Print the final matrix
#print(mat.od.death)

# Record the end time
# end_time <- Sys.time()

# Calculate and print the runtime
#runtime <- end_time - start_time
#print(runtime)

remainder <- job_id %% 100
result <- ifelse(remainder == 0, 100, remainder)

# Write the Matrix to a CSV file
output_file <- paste0("naloxone_iteration_", iteration,"_param_", result, ".csv")
write.csv(mat.od.death, file = output_file, row.names = TRUE)

print(paste("Job", job_id, "completed. Results saved to", output_file))