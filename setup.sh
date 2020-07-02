# exit when any command fails
set -e

# keep track of the last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
# echo an error message before exiting
# trap 'echo "\"${last_command}\" command filed with exit code $?."' EXIT

echo "Installing python requirements"
printf "\n"
pip install -r ./docs/requirements.txt

printf "\n"
echo "Creating necessary directories"
python ./tools/create_directories.py
printf "\n"

printf "\n"
echo "Compiling C code"
printf "\n"
gcc -fPIC -shared -o ./models/doubly_constrained/flow_forward_models.so ./models/doubly_constrained/flow_forward_models.c -O3
gcc -fPIC -shared -o ./models/doubly_constrained/potential_function.so ./models/doubly_constrained/potential_function.c -O3
gcc -fPIC -shared -o ./models/singly_constrained/potential_function.so ./models/singly_constrained/potential_function.c -O3

printf "\n"
echo "Generating stopping times"
printf "\n"

echo "Creating stopping times for each dataset"
datasets=(`cat ./docs/datasets.txt`)
noofelements=${#datasets[*]}
# Traverse the array
counter=0
while [ $counter -lt $noofelements ]
do
    python ./tools/stopping.py -data ${datasets[$counter]}
    echo "python ./tools/stopping.py -data ${datasets[$counter]}"
    counter=$(( $counter + 1 ))
done
# printf "\n"

# echo "Run tests"
# printf "\n"
