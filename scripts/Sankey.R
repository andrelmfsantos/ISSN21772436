# https://www.r-graph-gallery.com/321-introduction-to-interactive-sankey-diagram-2.html
# https://stackoverflow.com/questions/44132423/creating-a-sankey-diagram-using-networkd3-package-in-r

setwd("C:/Users/andre/OneDrive/EDUCAÇÃO/UNINOVE/Doutorado_ADM_2020/ARTIGOS_EM_DESENVOLVIMENTO/Artigo_OpenInnovation-TopicModel") # muda a pasta de destino

# Library
library(networkD3)
library(dplyr)

# A connection data frame is a list of flows with intensity for each flow

df <- read.csv2(file = "sankey.csv")
str(df)
head(df)
df <- filter(df, Quartile == "Q1")
#df <- filter(df, Country_Journal %in% c("Argentina", "Brazil", "Chile", "Venezuela"))
links <- data.frame(source = df$Source_Paper,target = df$Country_Journal)
links$count <- 1
links <- links %>% group_by(source, target) %>% summarise(value = sum(count))
links
nrow(links)

# From these flows we need to create a node data frame: it lists every entities involved in the flow
nodes <- data.frame(name = unique(c(links$source, links$target)))

links$IDsource <- match(links$source, nodes$name)-1 
links$IDtarget <- match(links$target, nodes$name)-1

# Make the Network
p <- sankeyNetwork(Links = links, Nodes = nodes,
                   Source = "IDsource", Target = "IDtarget",
                   Value = "value", NodeID = "name", 
                   sinksRight=FALSE, fontSize = 12)
p
