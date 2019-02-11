files<- c("sigma_u.feather", "sigma_v.feather", "sigma_theta.feather", "covar_from_kernel_on_train.feather", "M_one.feather", "adjusted_R.feather","y_data.feather","X_data.feather","noise.feather")

rfiles<- c("sigma_u.RData", "sigma_v.RData", "sigma_theta.RData", "covar_from_kernel_on_train.RData", "M_one.RData", "adjusted_R.RData","y_data.RData","X_data.RData","noise.RData")

for (i in 1:9){
	
	path <- files[i]
	df <- read_feather(path)
	save(df, file = rfiles[i])
	
	
}

