#!/bin/bash 
#
# in order to run this script you'd first need to build the ckProfiler executable in ../build/bin/
# run the script as "./run_gemm_performance_tests.sh <verification> <tag for your test environment> <branch name> <node name> <arch>
# input arguments: 
# verification = 0 : do not verify result correctness on CPU
#              = 1 : verify correctness on CPU (may take a long time)
# environment tag  : a string describing the specifics of your test environment
# branch name      : name of the branch in git repo (git status | grep -e 'On branch')
# node name        : $hostname
# arch             : GPU architecture, e.g. "gfx9" or "gfx1100"

#get the command line arguments:
export verify=$1
echo 'Verification: ' $verify
export env_type=$2
echo 'Environment type: ' $env_type
export branch=$3
echo 'Branch name: ' $branch
export host_name=$4
echo 'Host name: ' $host_name
export arch=$5
echo 'GPU architecture: ' $arch

function print_log_header(){
	rm -f $1;
	echo 'On branch ' $3 &> $1;
	echo 'Node name: ' $4 >> $1;
	#get GPU_arch and number of compute units from rocminfo
	echo -n "GPU_arch: " >> $1; rocminfo | grep "Name:" | grep "gfx" >> $1;
	rocminfo | grep "Compute Unit:" >> $1;
	hipcc --version | grep -e 'HIP version'  >> $1;
	echo 'Environment type: ' $2 >> $1;
	/opt/rocm/bin/amdclang++ --version | grep -e 'InstalledDir' >> $1;
}

#run ONNX gemm tests
export onnx_log="perf_onnx_gemm_$arch.log"
print_log_header $onnx_log $env_type $branch $host_name
./profile_onnx_gemm.sh gemm 0 0 $verify 1 0 1 2>&1 | tee -a $onnx_log
./profile_onnx_gemm.sh gemm 1 0 $verify 1 0 1 2>&1 | tee -a $onnx_log
