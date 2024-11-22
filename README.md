
# Artifact for PPoPP'25 paper 
> *GLUMIN: Fast Connectivity Check Based on LUTs For Efficient Graph Pattern Mining.* 

## 1.1. Download datasets.

+ Download datasets.
Datasets are available [here](https://drive.google.com/drive/folders/1ZV5oeyJfCV922bwwKoWfe6SLwIoiaNOG?usp=sharing).  
+ `/dataset_table1` include all datasets shown in Table 1.
+ `/dataset_135` and `/dataset_table1` include all 135 datasets to test Figure 8.  


## 1.2. Compile implementation.
To build the project for your GPU, modify the CUDA_ARCH option in the common.mk file:`src/common.mk`. Make sure to choose the appropriate architecture based on your GPU.

```
mkdir bin && make
```


# 2. Run initial test experiment with basic APIs.
## 2.1. Run baseline G<sup>2</sup>Miner.
>Usage: ./bin/pattern_gpu_GM <graph_path> <pattern_name>  
>Support Patterns in Figure 7: P1-P24

```
./bin/pattern_gpu_GM ./datasets/cit-Patents/graph P1
./bin/pattern_gpu_GM ./datasets/youtube/graph P13
./bin/pattern_gpu_GM ./datasets/soc-pokec/graph P17
```

## 2.2. Run G<sup>2</sup>Miner + LUT.
>Usage: ./bin/pattern_gpu_GM_LUT <graph_path> <pattern_name>  
>Support Patterns in Figure 7: P1-P24

```
./bin/pattern_gpu_GM_LUT ./datasets/cit-Patents/graph P1
./bin/pattern_gpu_GM_LUT ./datasets/youtube/graph P13
./bin/pattern_gpu_GM_LUT ./datasets/soc-pokec/graph P17
```

## 2.3. Run GraphFold & GraphFold + LUT.
>Usage: ./bin/pattern_gpu_GF_LUT <graph_path> <pattern_name> [use_lut]  
>Support Patterns in Figure 7: P1, P5, P10, P13.  
+ `[use_lut]` If the string "lut" is provided, the program will enable LUT mode.

```
./bin/pattern_gpu_GF_LUT ./datasets/mico/graph P1
./bin/pattern_gpu_GF_LUT ./datasets/mico/graph P1 lut
```


## 2.4. Run AutoMine & AntoMine + LUT.
>Usage: ./bin/automine_LUT <graph_path> <pattern_name> [use_lut]  
>Support Patterns in Figure 7: P1, P7, P10, P13, P15, P20  
+ `[use_lut]` If the string "lut" is provided, the program will enable LUT mode.

```
./bin/automine_LUT ./datasets/mico/graph P1
./bin/automine_LUT ./datasets/mico/graph P1 lut
```

## 2.5. Codegen

Codegen source: `codegen/*`  
Codegen pattern yml example: `codegen/codegen/patterns`  
Codegen kernel example: `codegen/include/generated/generated.cuh`

## Codegen from 4-star.yml (P1) and Test
+ Codegen LUT kernel(build LUT in Level 1)
```
cd scripts && ./codegen.sh 1
```
+ Make and run
```
cd .. && make && cd scripts && ./run.sh
```
