#######################
## Author: chenzhuoyang
## Date: 2022-12-09
## Content: EDA and feature selection
######################

library(Seurat)
library(magrittr)
library(circlize)
library(ggplot2)
library(umap)

setwd("D:/WorkFile/HKUST/autoML/model2/data/")
dat <- read.csv("feature_processed.csv")
features <- dat[,2:1164]
info <- dat[,1165:1169]
info$class <- 0
for (i in 1:nrow(info)) {
  ss <- info$label_M[i]
  if (ss < 3.5) {
    info$class[i] <- 0
  } else if (ss < 4.0) {
    info$class[i] <- 1
  } else if (ss < 4.5) {
    info$class[i] <- 2
  } else if (ss < 5.0) {
    info$class[i] <- 3
  } else {
    info$class[i] <- 4
  }
}
info$class <- as.character(info$class)
info$area <- as.character(info$area)
table(info$class)
#    0    1    2    3    4 
# 1420   57   32   20   14


# calculations
features <- as.matrix(features)
dim(features) #1543 1163

colMax <- apply(features, 2, max)
colMin <- apply(features, 2, min)
colRange <- colMax - colMin
colMin <- matrix(rep(colMin, times=nrow(features)), nrow = nrow(features), byrow = T)
colRange <- matrix(rep(colRange, times=nrow(features)), nrow = nrow(features), byrow = T)

colMean <- colMeans(features)
colMean <- matrix(rep(colMean, times=nrow(features)), nrow = nrow(features), byrow = T)
colSD <- apply(features, 2, sd)
colSD <- matrix(rep(colSD, times=nrow(features)), nrow = nrow(features), byrow = T)

# normalized to 0-1
normalized <- (features - colMin) / colRange

SD <- apply(normalized, 2, sd)
range(SD)
png("SDminmax_hist.png", width = 5.5, height = 5, units = "in", res = 300)
hist(SD, main = "Histogram of s.d. for Min-Max Normalized features", xlab = "Standard Deviation")
dev.off()

# scales to z-scores
scaled <- (features - colMean) / colSD
colinear <- cor(scaled)

top_anno <- HeatmapAnnotation(type = gsub(".*_","",colnames(colinear)))
color_v=c("#fbb4ae","#b3cde3") #magn vs. sound
names(color_v)=c("magn","sound")
set.seed(667)
png("colinear.png", width = 10, height = 8, units = "in", res = 300)
Heatmap(colinear,
        col = colorRamp2(c(0, 1), c("white", "red")),
        show_column_names = F,show_row_names = F, show_row_dend = F, show_column_dend = F,
        heatmap_legend_param = list(title = "Pearson", direction = "vertical",
                                    title_position = "leftcenter-rot",at=c(0,1),legend_height = unit(3, "cm")),
        top_annotation = top_anno,
        row_title = NULL, column_title = "Colinearity of 1163 features", column_title_side = "top",
        column_title_gp = gpar(fontsize = 18, fontface = "bold"),
        #column_names_gp = grid::gpar(fontsize = columnFont),
        #column_names_rot = 45,
        #rect_gp = gpar(col = "white", lwd = gridBorder)
)
dev.off()

s = c()
for(i in 1:(ncol(features)-1)){
  s = c(s, colinear[i,(i+1):ncol(features)])
}

name1 = c()
for(i in 1:(ncol(features)-1)){
  name1 = c(name1, rep(colnames(features)[i],ncol(features)-i))
}

name2 = c()
for(i in 1:(ncol(features)-1)){
  name2 = c(name2, colnames(features)[(i+1):ncol(features)])
}

df_cor <- data.frame(feature1 = name1, feature2 = name2, cor = s)
png("colinear_hist.png", width = 4.5, height = 4.5, units = "in", res = 300)
hist(df_cor$cor, main = "Histogram of Correlation between Features", xlab = "Pearson Correlation")
dev.off()
summary(df_cor$cor)

df_cor$keep <- TRUE
for (i in 1:nrow(df_cor)) {
  if(df_cor$keep[i]){
    if(df_cor$cor[i]>0.8 | df_cor$cor[i]< -0.8) df_cor$keep[df_cor$feature1==df_cor$feature1[i]] = FALSE
  }
}
length(unique(df_cor$feature1[df_cor$keep])) #128

keepFeatures = rep(TRUE, ncol(features))
names(keepFeatures) = colnames(features)

sum(SD>0.1) #496
diag(colinear) = 0
Corpts <- apply(colinear, 2, function(x){sum(x>0.8|x< -0.8) / ncol(colinear)})
hist(Corpts)

keepFeatures[SD<0.1] = FALSE
keepFeatures[!names(keepFeatures)%in%unique(df_cor$feature1[df_cor$keep])] = FALSE
sum(keepFeatures) #38

##PCA + UMAP
pca <- prcomp(scaled)
plot(pca$x[,1], pca$x[,2])
ggplot(data.frame(PC1=pca$x[,1],PC2=pca$x[,2],area=info$area,event=info$class,set=info$flag),
       aes(x = PC1, y = PC2, size = event, color = area)) + 
  geom_point()

library(factoextra)
fviz_eig(pca, ncp = 20)
var<-get_pca_var(pca)
a<-fviz_contrib(pca, "var", axes=1, xtickslab.rt=45)
library(dplyr)
adata <- a$data %>% arrange(desc(contrib)) %>% top_n(10)
adata$name <- factor(adata$name, levels = adata$name)
ggplot(adata, aes(name, contrib)) + geom_bar(stat = "identity") + RotatedAxis()


umap_data <- umap(pca$x, components = 20, min_dist=0.2)
ggplot(data.frame(UMAP1=umap_data$layout[,1],UMAP2=umap_data$layout[,2],area=info$area,event=info$label_M,set=info$flag),
       aes(x = UMAP1, y = UMAP2, size = event, color = area)) + geom_point()
  geom_point()

## Conclusion: there is obvious clustering on area and magnitude class, which is
## indeed as variables in our models
  

  
## After removing
pca2 <- prcomp(scale(features[,keepFeatures]))
plot(pca2$x[,1], pca2$x[,2])
ggplot(data.frame(PC1=pca2$x[,1],PC2=pca2$x[,2],area=info$area,event=info$label_M,set=info$flag),
       aes(x = PC1, y = PC2, size = event, color = area)) + 
  geom_point()

fviz_eig(pca2, ncp = 20)
var<-get_pca_var(pca2)
a<-fviz_contrib(pca2, "var", axes=1, xtickslab.rt=45)
adata <- a$data %>% arrange(desc(contrib)) %>% top_n(10)
adata$name <- factor(adata$name, levels = adata$name)
ggplot(adata, aes(name, contrib)) + geom_bar(stat = "identity") + theme_classic(base_size = 10) + RotatedAxis() +
  theme(plot.margin=margin(20,50,20,100), plot.title = element_text(hjust = 0.5, size = 16)) + ggtitle("Features Contributions")
ggsave("feature_contrub_afterFilter.png", width = 6, height = 6)

#choose PCA num
adata <- a$data %>% arrange(desc(contrib)) %>% top_n(20)
sum(adata$contrib)

umap_data2 <- umap(pca2$x, components = 20, min_dist=0.2)
ggplot(data.frame(UMAP1=umap_data2$layout[,1],UMAP2=umap_data2$layout[,2],area=info$area,event=info$class,set=info$flag),
       aes(x = UMAP1, y = UMAP2, size = event, color = area)) + geom_point()
geom_point()
ggsave("UMAP_afterFilter.png", width = 6, height = 5)

## newdata
newdat <- cbind(Day=dat[,1],features[,keepFeatures], info[,-6], pca2$x[,1:5])
