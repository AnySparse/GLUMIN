#!/bin/bash

datasets=(mico)  # Dataset list
patterns=(P1 P7 P10 P13 P15 P20)  # Test pattern list

# Output files
summary_file="log/AutoMine_summary_table.log" # Runtime result table
log_file="log/AutoMine_output.log" # Execution output

column_width=20
column_width_1=8
column_width_2=13

echo -e "Runtime of AutoMine and AutoMine + LUT (sec)" > $summary_file
echo -e "+------------------------------------------------------------------------------------+" >> $summary_file

header="| Pattern  | Optimization "
for dataset in "${datasets[@]}"; do
  header+="| $(printf "%-${column_width}s" $dataset)"
done
echo "$header" >> $summary_file
echo -e "+------------------------------------------------------------------------------------+" >> $summary_file

echo -e "Execution log started at $(date)" > $log_file

for pattern in "${patterns[@]}"; do
  row="| $(printf "%-${column_width_1}s" $pattern) | $(printf "%-${column_width_2}s" "None")"
  
  for dataset in "${datasets[@]}"; do
    command="./bin/automine_LUT ./datasets/$dataset/graph $pattern"
    
    output=$($command 2>&1)
    
    echo -e "\n====================\nRunning command: $command\n" >> $log_file
    echo "$output" >> $log_file
    
    runtime=$(echo "$output" | grep -oP "runtime \[Automine\] = \K\d+\.\d+")
    count=$(echo "$output" | grep -oP "Pattern \S+ count: \K\d+")
    
    if [[ -n "$runtime" && -n "$count" ]]; then
      row+="| $(printf "%-${column_width}.6f" $runtime)" 
    else
      row+="| $(printf "%-${column_width}s" "Error")"
    fi
  done

  echo "$row" >> $summary_file

  row_lut="| $(printf "%-${column_width_1}s" $pattern) | $(printf "%-${column_width_2}s" "LUT")"
  
  for dataset_lut in "${datasets[@]}"; do
    command_lut="./bin/automine_LUT ./datasets/$dataset_lut/graph $pattern lut"
    
    output_lut=$($command_lut 2>&1)
    
    echo -e "\n====================\nRunning command: $command_lut\n" >> $log_file
    echo "$output_lut" >> $log_file
    
    runtime_lut=$(echo "$output_lut" | grep -oP "runtime \[Automine \+ LUT\] = \K\d+\.\d+")
    count_lut=$(echo "$output_lut" | grep -oP "Pattern \S+ count: \K\d+")
    
    if [[ -n "$runtime_lut" && -n "$count_lut" ]]; then
      row_lut+="| $(printf "%-${column_width}.6f" $runtime_lut)"
    else
      row_lut+="| $(printf "%-${column_width}s" "Error")"
    fi
  done

  echo "$row_lut" >> $summary_file

done

echo -e "+------------------------------------------------------------------------------------+" >> $summary_file

echo -e "\nExecution completed at $(date)" >> $log_file
