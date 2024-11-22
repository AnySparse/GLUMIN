ROOT_DIR=$(realpath $(dirname $0))/..
PAT_DIR=${ROOT_DIR}"/codegen/patterns"
LOG_DIR=${ROOT_DIR}"/scripts/logs/"$(date +%s)/
mkdir -p ${LOG_DIR}
CODEGEN_DIR=${ROOT_DIR}"/codegen"
SCRIPT_DIR=${ROOT_DIR}"/scripts"

TEMPLATE=${ROOT_DIR}"/include/gpu_kernels/template.cuh"
GENERATED=${ROOT_DIR}"/include/generated/generated.cuh"

for file in ${PAT_DIR}/*; do
  if [ -f "$file" ]; then
    echo $(basename $file)
  fi
  n=$(echo $file | sed -n 's/[^0-9]*\([0-9]\).*/\1/p')
  for ((i=0; i<n-1; i++)); do
    python3 ${CODEGEN_DIR}/codegen.py -e -p $file -l $i -t ${TEMPLATE} -n generated_kernel -c ${GENERATED} |& tee ${LOG_DIR}$(basename $file).codegen_$i
    cd ${ROOT_DIR}
    make clean
    make |& tee ${LOG_DIR}$(basename $file).make_$i
    cd -
    cd ${SCRIPT_DIR}
    timeout 30 ./run.sh |& tee ${LOG_DIR}$(basename $file).run_$i
    cd -
  done
done
