# This script is made to work with other scripts from KI_project available on
# Github (https://github.com/rickettsi/KI_project)

setwd("~/DATA")
library(dplyr)
library(ggtree)
library(treeio)
library(ggplot2)
library(ggforce)
library(ape)
library(stringr)
library(ggtreeExtra)
library(ggimage)

tree <- read.tree("tree.nwk")
AnceState_final <- read.csv("AnceState_final.csv")

p <- ggtree(tree)
tree_data <- p$data

tip_state <- str_sub(tree$tip.label, -1)
states <- data.frame(
  node = seq_along(tip_state),
  state = tip_state)

states$state <- as.numeric(states$state)

node_sorted <- tree_data %>%
  filter(isTip == F) %>%
  arrange(x) %>%
  select(node)
print(node_sorted)

AnceState_final$state <- as.numeric(AnceState_final$state)
AnceState_final$node <- node_sorted$node
print(AnceState_final)

AllStates <- bind_rows(states, AnceState_final)
AllStates$state <- as.factor(AllStates$state)
AllStates <- AllStates %>% arrange(node)
print(AllStates)

tree_all_data <- inner_join(AllStates,tree_data,by='node')

ggtree(tree_all_data) +
  #geom_tiplab() +
  geom_point(aes(color=state), shape=15, size=3)+
  scale_color_manual(
    name = "State",
    values = c("0" =  "#C07CA4", "1" = "#04883a")) +
  theme_tree()+
  theme(legend.position="top",legend.background = element_rect(fill = "lightgray"))
ggsave(filename = "true_tree.pdf", dpi = 600)

##############################################
nb_tips <- length(tree$tip.label)

state_colors <- c("Pstate_0" = "#C07CA4", "Pstate_1" = "#04883a")

node_pie_data <- AllStates %>%
  filter(node > nb_tips) %>%
  select(node, Pstate_0, Pstate_1)


pies <- nodepie(node_pie_data, cols = 2:3, color = NA)  # Pas de contour
for (i in seq_along(pies)) {
  pies[[i]] <- pies[[i]] + scale_fill_manual(values = state_colors)
}
################### PLOT ############
p5 <- ggtree(tree) %<+% AllStates +
  geom_tippoint(aes(color = state),shape = 15, size = 4,show.legend=TRUE) +
  scale_color_manual(
    name = "State",
    breaks = c("0", "1"),
    values = c("0" = "#C07CA4", "1" = "#04883a"),
    drop = FALSE)+
  theme(legend.position="top",legend.background = element_rect(fill = "lightgray"))
p5

p5 <- inset(p5, pies, width = 0.03, height = 0.03, hjust = 0.005)
p5
ggsave(filename = "reconstructed_tree.pdf", dpi = 600)

#####################################################
rep_try <- read.csv("SimNtrees2.csv")
library(WVPlots)
rep_try <- rep_try[1:5000,]
ScatterHist(
  rep_try,
  "n_state1",
  "n_state2",
  title = "Simulated tree size distribution",
  smoothmethod = "gam",
  contour = TRUE,
  point_color = "#006d2c",
  # dark green
  hist_color = "#6baed6",
  # medium blue
  smoothing_color = "#54278f",
  # dark purple
  density_color = "#08519c",
  # darker blue
  contour_color = "#9e9ac8"
) # lighter purple

