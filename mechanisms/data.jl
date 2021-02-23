using NPZ


#
# download data files into the data directory from here:
# https://github.com/dpcomp-org/dpcomp_core/tree/master/dpcomp_core/datafiles
#

getincome() = npzread("data/INCOME.n4096.npy")
getnet() = npzread("data/NETTRACE.n4096.npy")
getsearch() = npzread("data/SEARCHLOGS.n4096.npy")
