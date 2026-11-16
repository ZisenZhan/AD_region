Sys.setlocale('LC_CTYPE','English_United States.utf8')
O<-normalizePath(commandArgs(TRUE)[1],winslash='/');B<-dirname(dirname(dirname(O)))
sc<-file.path(B,'单细胞补充验证_1023')
.libPaths(c(file.path(sc,'环境/R_libs'),.libPaths()))
suppressPackageStartupMessages({library(edgeR);library(limma)})
prior<-file.path(B,'脑区功能失调模式与单核呼应_20261113')
read<-function(p)read.csv(if(grepl('gz$',p))gzfile(p)else p,fileEncoding='UTF-8-BOM',check.names=FALSE)
m<-read(file.path(prior,'单核结果/样本纳入排除.csv.gz'))
m<-subset(m,dataset=='SEAAD'&kind=='regional'&phenotype=='local_CPS'&region=='frontal_prefrontal'&included==TRUE)
maps<-read(file.path(prior,'输入/单核基因名称映射.csv'))
spec<-read(file.path(O,'机制清单.csv'));members<-read(file.path(O,'额叶方向模板.csv'))
outs<-list()
for(ct in unique(m$cell_type)){
  k<-subset(m,cell_type==ct)
  f<-file.path(sc,'结果/SEAAD_准备/counts',paste0(ct,'.csv.gz'))
  hdr<-names(read.csv(gzfile(f),nrows=0,check.names=FALSE))
  stopifnot(all(k$sample_id%in%hdr),!anyDuplicated(k$donor_id),all(k$n_nuclei>=20),length(unique(k$CPS_Local))>=6,min(table(k$group))>=5)
  cl<-ifelse(hdr%in%k$sample_id,'numeric','NULL');cl[1]<-'character'
  raw<-read.csv(gzfile(f),check.names=FALSE,colClasses=cl)
  nm<-maps$canonical[match(raw[[1]],maps$original)];nm[is.na(nm)]<-raw[[1]][is.na(nm)]
  mat<-rowsum(as.matrix(raw[,k$sample_id,drop=FALSE]),nm,reorder=FALSE)
  stopifnot(all(is.finite(mat)),all(mat>=0),all(mat==floor(mat)),identical(colnames(mat),k$sample_id))
  des<-model.matrix(~scale(CPS_Local),k); y<-DGEList(mat)
  y<-calcNormFactors(y[filterByExpr(y,design=des),,keep.lib.sizes=FALSE]);v<-voom(y,des,plot=FALSE)
  idx<-lapply(spec$go_id,function(g)which(rownames(v$E)%in%members$gene[members$go_id==g]));names(idx)<-spec$go_id
  weight<-members$weight[match(rownames(v$E),members$gene)];weight[is.na(weight)]<-1
  good<-lengths(idx)>=10
  tab<-data.frame(cell_type=ct,go_id=spec$go_id,template_n=lengths(idx),donors=nrow(k),p=NA_real_,direction=NA_character_)
  if(any(good)){
    res<-fry(v,idx[good],des,contrast=2,gene.weights=weight,sort='none')
    j<-match(tab$go_id,rownames(res));tab$p<-res$PValue[j];tab$direction<-as.character(res$Direction[j])
  }
  outs[[ct]]<-tab
  cat(ct,nrow(k),sum(good),'\n');flush.console()
}
z<-do.call(rbind,outs);stopifnot(length(outs)==8)
z$fdr<-p.adjust(z$p,'BH',n=nrow(z))
write.csv(z,file.path(O,'六机制八类细胞检验.csv'),row.names=FALSE,na='')
write.csv(m,file.path(O,'实际纳入供体.csv'),row.names=FALSE)
