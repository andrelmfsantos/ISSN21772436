#====================================================
# Author: André Luis M.F. dos Santos
# e-mail: andrelmfsantos@gmail.com
# Title: Chord Diagram
# Date: 2021 / Feb / 02
# Reference: https://www.r-graph-gallery.com/123-circular-plot-circlize-package-2.html
# Reference: https://jokergoo.github.io/circlize_book/book/advanced-usage-of-chorddiagram.html
#====================================================

# I. Entrada dos dados

# Diretórios e Arquivos:
# getwd() # Qual o diretório que o script está apontando
# list.files() # Quais arquivos estão contidos no diretório
setwd("C:/Users/andre/OneDrive/EDUCAÇÃO/UNINOVE/Doutorado_ADM_2020/ARTIGOS_EM_DESENVOLVIMENTO/Artigo_OpenInnovation-TopicModel/OSF_OpenInnovation") # muda a pasta de destino
# Leitura de uma base externa
df <- read.csv2(file = "ChordMatrix.csv")  # Cria data frame do CSV
row.names(df) <- df$ï..Subject        # Primeira coluna como index
df$ï..Subject <- NULL                   # Exclui a primeira coluna que agora é um index
str(df)
mat <- data.matrix(df)
str(mat)
print(mat)

# Load the circlize library
#install.packages("circlize")
library(circlize)

rand_color(19)
grid.col = c(S03 = "#AEADD1FF", S05 = "#68852AFF", S07 = "#AB7D5FFF", S08 = "#9EE229FF", 
             S09 = "#494835FF", S10 = "#0C0703FF", S11 = "#C88C69FF", S12 = "#3D7E4DFF",
             S13 = "#6B6F4AFF", S14 = "#81AD57FF", S15 = "#585364FF", S16 = "#0D4A32FF", 
             S18 = "#3C9D85FF", S19 = "#010502FF", S20 = "#68623BFF", S21 = "#040707FF",
             S24 = "#DCABC3FF", S26 = "#420391FF", S27 = "#F79BA4FF")

chordDiagram(mat)
# Make the circular plot
chordDiagram(mat, transparency = 0.5)
#-------------------------------------------------
# 15.2 Customize sector labels
chordDiagram(mat, grid.col = grid.col, annotationTrack = "grid", 
             preAllocateTracks = list(track.height = max(strwidth(unlist(dimnames(mat))))))
# we go back to the first track and customize sector labels
circos.track(track.index = 1, panel.fun = function(x, y) {
  circos.text(CELL_META$xcenter, CELL_META$ylim[1], CELL_META$sector.index, 
              facing = "clockwise", niceFacing = TRUE, adj = c(0, 0.5), fontSize = 10)
}, bg.border = NA) # here set bg.border to NA is important
##############################
chordDiagram(mat, grid.col = grid.col, 
             annotationTrack = c("grid", "axis"), annotationTrackHeight = mm_h(5))
for(si in get.all.sector.index()) {
  xlim = get.cell.meta.data("xlim", sector.index = si, track.index = 1)
  ylim = get.cell.meta.data("ylim", sector.index = si, track.index = 1)
  circos.text(mean(xlim), mean(ylim), si, sector.index = si, track.index = 1, 
              facing = "bending.inside", niceFacing = TRUE, col = "white")
}
#------------------------
#par(mfrow = c(1, 1))
