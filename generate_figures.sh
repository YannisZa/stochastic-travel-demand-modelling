# exit when any command fails
set -e

# keep track of the last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG

# First argument is the figure number
figure_number="$1"
dataset="$2"
mode="$3" # s for stochastic and d for deterministic


# if [[ $figure_number = "4a" ]] || [[ -z $figure_number ]]; then
#   echo "-------------Generating data and plots for potential grid search estimation-------------"
#
#   if [[ $dataset = "synthetic" ]] || [[ -z $dataset ]]; then
#     echo "$dataset = synthetic"
#     if [[ $mode = "d" ]] || [[ -z $mode ]]; then
#       echo "mode = deterministic"
#       python visualisation/potential_analysis.py -data synthetic -d 0 -bmax 2800000 -amin -2 -n 100 -hide
#     fi
#     if [[ $mode = "s" ]] || [[ -z $mode ]]; then
#       echo "mode = stochastic"
#       python visualisation/potential_analysis.py -data synthetic -d 0.00617494255748621 -bmax 10 -amax 0.2 -n 100 -hide
#     fi
#   fi
#
#   if [[ $dataset = "retail" ]] || [[ -z $dataset ]]; then
#     echo "$dataset = retail"
#     if [[ $mode = "d" ]] || [[ -z $mode ]]; then
#       echo "mode = deterministic"
#       python visualisation/potential_analysis.py -data retail -d 0 -bmax 1400000 -n 1000 -hide
#     fi
#     if [[ $mode = "s" ]] || [[ -z $mode ]]; then
#       echo "mode = stochastic"
#       python visualisation/potential_analysis.py -data retail -d 0.00617494255748621 -bmax 1400000 -n 1000 -hide
#     fi
#   fi
#
#   if [[ $dataset = "commuter_ward" ]] || [[ -z $dataset ]]; then
#     echo "$dataset = commuter_ward"
#     if [[ $mode = "d" ]] || [[ -z $mode ]]; then
#       echo "mode = deterministic"
#       python visualisation/potential_analysis.py -data commuter_ward -d 0 -bmax 1000000 -n 1000 -hide
#     fi
#     if [[ $mode = "s" ]] || [[ -z $mode ]]; then
#       echo "mode = stochastic"
#       python visualisation/potential_analysis.py -data commuter_ward -d 0.01178781925343811 -bmax 1000000 -n 1000 -hide
#     fi
#   fi
#
#   if [[ $dataset = "commuter_borough" ]] || [[ -z $dataset ]]; then
#     echo "$dataset = commuter_borough"
#     if [[ $mode = "d" ]] || [[ -z $mode ]]; then
#       echo "mode = deterministic"
#       python visualisation/potential_analysis.py -data commuter_borough -d 0 -bmax 1000000 -n 1000 -hide
#     fi
#     if [[ $mode = "s" ]] || [[ -z $mode ]]; then
#       echo "mode = stochastic"
#       python visualisation/potential_analysis.py -data commuter_borough -d 0.01178781925343811 -bmax 1000000 -n 1000 -hide
#     fi
#   fi
#   echo "----------------------------------------------------------------------------------------------------"
#   printf "\n"
# fi


if [[ $figure_number = "4b" ]] || [[ -z $figure_number ]]; then
  echo "-------------Generating data and plots for R^2 grid search estimation-------------"

  if [[ $dataset = "synthetic" ]] || [[ -z $dataset ]]; then
    echo "$dataset = synthetic"
    if [[ $mode = "d" ]] || [[ -z $mode ]]; then
      echo "mode = deterministic"
      python inference/rsquared_analysis.py -data synthetic -d 0 -bmax 1400000 -n 100 -hide
    fi
    if [[ $mode = "s" ]] || [[ -z $mode ]]; then
      echo "mode = stochastic"
      python inference/rsquared_analysis.py -data synthetic -d 0.00617494255748621 -bmax 1400000 -n 100 -hide
    fi
  fi

  if [[ $dataset = "retail" ]] || [[ -z $dataset ]]; then
    echo "$dataset = retail"
    if [[ $mode = "d" ]] || [[ -z $mode ]]; then
      echo "mode = deterministic"
      python inference/rsquared_analysis.py -data retail -d 0 -bmax 1400000 -n 1000 -hide
    fi
    if [[ $mode = "s" ]] || [[ -z $mode ]]; then
      echo "mode = stochastic"
      python inference/rsquared_analysis.py -data retail -d 0.00617494255748621 -bmax 1400000 -n 1000 -hide
    fi
  fi

  if [[ $dataset = "commuter_ward" ]] || [[ -z $dataset ]]; then
    echo "$dataset = commuter_ward"
    if [[ $mode = "d" ]] || [[ -z $mode ]]; then
      echo "mode = deterministic"
      python inference/rsquared_analysis.py -data commuter_ward -d 0 -bmax 1000000 -n 1000 -hide
    fi
    if [[ $mode = "s" ]] || [[ -z $mode ]]; then
      echo "mode = stochastic"
      python inference/rsquared_analysis.py -data commuter_ward -d 0.01178781925343811 -bmax 1000000 -n 1000 -hide
    fi
  fi

  if [[ $dataset = "commuter_borough" ]] || [[ -z $dataset ]]; then
    echo "$dataset = commuter_borough"
    if [[ $mode = "d" ]] || [[ -z $mode ]]; then
      echo "mode = deterministic"
      python inference/rsquared_analysis.py -data commuter_borough -d 0 -bmax 1000000 -n 1000 -hide
    fi
    if [[ $mode = "s" ]] || [[ -z $mode ]]; then
      echo "mode = stochastic"
      python inference/rsquared_analysis.py -data commuter_borough -d 0.01178781925343811 -bmax 1000000 -n 1000 -hide
    fi
  fi
  echo "----------------------------------------------------------------------------------------------------"
  printf "\n"
fi

if [[ $figure_number = "4c" ]] || [[ -z $figure_number ]]; then
  echo "-------------Generating data and plots for log-likelihood grid search estimation using Laplace approximation-------------"

  if [[ $dataset = "retail" ]] || [[ -z $dataset ]]; then
    echo "$dataset = retail"
    if [[ $mode = "d" ]] || [[ -z $mode ]]; then
      echo "mode = deterministic"
      python inference/laplace_analysis.py -data retail -d 0.00617494255748621 -bmax 1400000 -g 100 -hide
    fi
    if [[ $mode = "s" ]] || [[ -z $mode ]]; then
      echo "mode = stochastic"
      python inference/laplace_analysis.py -data retail -d 0.00617494255748621 -bmax 1400000 -g 10000 -hide
    fi
  fi

  # python inference/laplace_analysis.py -data synthetic -d 0.2727272727272727 -bmax 12 -g 100
  # python inference/laplace_analysis.py -data synthetic -d 0.2727272727272727 -bmax 12 -g 10000


  if [[ $dataset = "commuter_ward" ]] || [[ -z $dataset ]]; then
    echo "$dataset = commuter_ward"
    if [[ $mode = "d" ]] || [[ -z $mode ]]; then
      echo "mode = deterministic"
      python inference/laplace_analysis.py -data commuter_ward -d 0.01178781925343811 -bmax 1400000 -g 10000 -hide # or bmax 71260
    fi
    if [[ $mode = "s" ]] || [[ -z $mode ]]; then
      echo "mode = stochastic"
      python inference/laplace_analysis.py -data commuter_ward -d 0.01178781925343811 -bmax 1400000 -g 100 -hide# or bmax 71260
    fi
  fi

  if [[ $dataset = "commuter_borough" ]] || [[ -z $dataset ]]; then
    echo "$dataset = commuter_borough"
    if [[ $mode = "d" ]] || [[ -z $mode ]]; then
      echo "mode = deterministic"
      python inference/laplace_analysis.py -data commuter_borough -d 0.01178781925343811 -bmax 1400000 -g 10000 -hide # or bmax 71260
    fi
    if [[ $mode = "s" ]] || [[ -z $mode ]]; then
      echo "mode = stochastic"
      python inference/laplace_analysis.py -data commuter_borough -d 0.01178781925343811 -bmax 1400000 -g 100 -hide # or bmax 71260
    fi
  fi

  echo "----------------------------------------------------------------------------------------------------"
  printf "\n"
fi

if [[ $figure_number = "6" ]] || [[ -z $figure_number ]]; then
  echo "-------------Generating data for latent posterior optimisation-------------"

  if [[ $dataset = "retail" ]] || [[ -z $dataset ]]; then
    echo "$dataset = retail"
    python inference/optimise_latent_posterior.py -data retail -d 0.00617494255748621 -a 0.5 -b 210000 -g 10000 -hide
    python inference/optimise_latent_posterior.py -data retail -d 0.00617494255748621 -a 1.0 -b 210000 -g 10000 -hide
    python inference/optimise_latent_posterior.py -data retail -d 0.00617494255748621 -a 1.5 -b 210000 -g 10000 -hide
    python inference/optimise_latent_posterior.py -data retail -d 0.00617494255748621 -a 2.0 -b 210000 -g 10000 -hide
  fi

  if [[ $dataset = "commuter_ward" ]] || [[ -z $dataset ]]; then
    echo "$dataset = commuter_ward"
    python inference/optimise_latent_posterior.py -data commuter_ward -d 0.01178781925343811 -a 0.5 -b 210000 -g 10000 -hide
    python inference/optimise_latent_posterior.py -data commuter_ward -d 0.01178781925343811 -a 1.0 -b 210000 -g 10000 -hide
    python inference/optimise_latent_posterior.py -data commuter_ward -d 0.01178781925343811 -a 1.5 -b 210000 -g 10000 -hide
    python inference/optimise_latent_posterior.py -data commuter_ward -d 0.01178781925343811 -a 2.0 -b 210000 -g 10000 -hide
  fi

  if [[ $dataset = "commuter_borough" ]] || [[ -z $dataset ]]; then
    echo "$dataset = commuter_borough"
    python inference/optimise_latent_posterior.py -data commuter_borough -d 0.01178781925343811 -a 0.5 -b 210000 -g 10000 -hide
    python inference/optimise_latent_posterior.py -data commuter_borough -d 0.01178781925343811 -a 1.0 -b 210000 -g 10000 -hide
    python inference/optimise_latent_posterior.py -data commuter_borough -d 0.01178781925343811 -a 1.5 -b 210000 -g 10000 -hide
    python inference/optimise_latent_posterior.py -data commuter_borough -d 0.01178781925343811 -a 2.0 -b 210000 -g 10000 -hide
  fi

  echo "---------------------------------------------------------------------------"
fi
