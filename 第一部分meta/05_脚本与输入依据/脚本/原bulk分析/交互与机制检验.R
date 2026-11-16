args <- commandArgs(trailingOnly=TRUE)
root <- normalizePath('.', winslash='/')
out <- normalizePath(args[1], winslash='/')
Sys.setlocale('LC_CTYPE','English_United States.utf8')
lib <- file.path(root,'outputs/ad_region_analysis/单细胞分析总库/单细胞补充验证_1023/环境/R_libs')
.libPaths(c(lib,.libPaths()))
suppressPackageStartupMessages(library(limma))
# Import only function definitions, never execute the historical output-writing driver.
expressions <- parse(file.path(root,'region_ad_analysis/29b_run_pairwise_region_disease_interaction.R'))
for(e in expressions) {
  if(is.call(e) && as.character(e[[1]])=='<-' && length(e)==3 && is.call(e[[3]]) && as.character(e[[3]][[1]])=='function') eval(e)
}
inputs <- list.files(file.path(out,'交互输入'),pattern='_meta.csv$',full.names=TRUE)
families <- list(GSE118553=TRUE,GSE48350=FALSE,GSE36980=FALSE,GSE122063=TRUE)
records <- list(); audits <- list()
for(f in names(families)) {
  paths <- inputs[grepl(paste0('__',f,'_meta.csv$'),inputs)]
  objects <- list()
  for(p in paths) {
    m <- read.csv(p,check.names=FALSE,stringsAsFactors=FALSE,fileEncoding='UTF-8-BOM')
    region <- m$region_slug[1]
    y <- as.matrix(read.csv(sub('_meta.csv$','_expression.csv.gz',p),row.names=1,check.names=FALSE))
    storage.mode(y) <- 'numeric'
    stopifnot(identical(colnames(y),as.character(m$donor_id)),!anyDuplicated(m$donor_id))
    m$sample_id <- paste(region,m$donor_id,sep='|');colnames(y) <- m$sample_id
    objects[[region]] <- list(matrix=y,metadata=m)
  }
  pairs <- combn(names(objects),2,simplify=FALSE)
  for(pair in pairs) {
    cache <- file.path(out,'交互结果',paste(f,pair[1],pair[2],'fit.rds',sep='___'))
    if(file.exists(cache)) {
      fit<-readRDS(cache);records[[length(records)+1]]<-fit
      audits[[length(audits)+1]]<-unique(fit[,c('family','pair','n_shared_donors','n_ad','n_control','model_type','covariates')])
      next
    }
    a<-objects[[pair[1]]];b<-objects[[pair[2]]]
    common<-intersect(rownames(a$matrix),rownames(b$matrix))
    keep<-vapply(common,function(g) all(is.finite(a$matrix[g,])) && all(is.finite(b$matrix[g,])),logical(1))
    common<-common[keep];a$matrix<-a$matrix[common,,drop=FALSE];b$matrix<-b$matrix[common,,drop=FALSE]
    map<-data.frame(feature_id=common,gene_symbol=common)
    fit<-if(families[[f]]) fit_paired_interaction(pair[1],pair[2],a,b,map) else fit_independent_interaction(pair[1],pair[2],a,b,map)
    fit$family<-f
    saveRDS(fit,cache)
    records[[length(records)+1]]<-fit
    audits[[length(audits)+1]]<-unique(fit[,c('family','pair','n_shared_donors','n_ad','n_control','model_type','covariates')])
    message('INTERACTION ',f,' ',paste(pair,collapse=' vs '),' genes=',nrow(fit))
  }
}
columns<-Reduce(union,lapply(records,names))
records<-lapply(records,function(d) {for(n in setdiff(columns,names(d))) d[[n]]<-NA;d[,columns]})
interaction <- do.call(rbind,records)
write.csv(interaction,gzfile(file.path(out,'交互结果/新输入重新拟合基因交互.csv.gz'),'wt'),row.names=FALSE)
write.csv(do.call(rbind,audits),file.path(out,'交互结果/新拟合交互设计核查.csv'),row.names=FALSE)
members <- read.csv(gzfile(file.path(out,'输入/六机制基因成员.csv.gz')),stringsAsFactors=FALSE)
sets <- split(members$gene_symbol,members$mechanism)
old <- read.csv(gzfile(file.path(out,'交互结果/原模型规范基因交互.csv.gz')),stringsAsFactors=FALSE)
# HBTRC matched 230-donor PFC is an interaction-only subset, not a new independent meta cohort.
old <- old[old$family=='HBTRC',]
common <- intersect(names(interaction),names(old))
all <- rbind(interaction[,common],old[,common])
write.csv(all,gzfile(file.path(out,'交互结果/最新完整基因交互.csv.gz'),'wt'),row.names=FALSE)
cam <- list()
test_sets <- function(stat,genes,scope,pair='') {
  index<-lapply(sets,function(gs) which(genes %in% gs)); index<-index[lengths(index)>=10]
  z<-cameraPR(stat,index,inter.gene.cor=0.01,sort=FALSE)
  z$mechanism<-rownames(z);z$scope<-scope;z$pair<-pair;z$correlation_assumption<-0.01
  s<-cameraPR(stat,index,inter.gene.cor=0.05,sort=FALSE)
  z$sensitivity_P_005<-s$PValue;z$sensitivity_FDR_005<-s$FDR
  rownames(z)<-NULL;z
}
for(f in unique(all$family)) for(p in unique(all$pair[all$family==f])) {
  d<-all[all$family==f & all$pair==p,];cam[[length(cam)+1]]<-test_sets(d$statistic,d$gene_symbol,f,p)
}
write.csv(do.call(rbind,cam),file.path(out,'交互结果/六机制cameraPR队列交互.csv'),row.names=FALSE)
cam<-list()
for(p in list.files(file.path(out,'结果'),pattern='_基因Meta.csv.gz$',full.names=TRUE)) {
  d<-read.csv(gzfile(p),stringsAsFactors=FALSE);scope<-sub('_基因Meta.csv.gz$','',basename(p))
  cam[[length(cam)+1]]<-test_sets(d$effect/d$se,d$gene_symbol,scope)
}
write.csv(do.call(rbind,cam),file.path(out,'结果/六机制cameraPR区域检验.csv'),row.names=FALSE)
writeLines(capture.output(sessionInfo()),file.path(out,'核对/R运行环境.txt'))
writeLines('completed',file.path(out,'核对/R阶段完成.txt'))
