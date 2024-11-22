cd ../codegen
python3 codegen.py -p patterns/4-star.yml -l $1 -t ../include/gpu_kernels/template.cuh -n generated_kernel -c ../include/generated/generated.cuh
