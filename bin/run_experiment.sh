source $(pwd)/.venv/bin/activate
export PYTHON=$(pwd)/.venv/bin/python

echo "Experiments in $1:"
for dir in $1/*; do
  if [ -f "$dir/main.py" ] && [ -f "$dir/config.yaml" ]
  then echo $dir
  fi
done

echo "\n"
for dir in $1/*; do
  if [ -f "$dir/main.py" ] && [ -f "$dir/config.yaml" ]
  then
    echo "run $PYTHON $(pwd)/$dir/main.py -c $(pwd)/$dir/config.yaml"
    $PYTHON "$(pwd)/$dir/main.py" -c "$(pwd)/$dir/config.yaml"
    echo "\n"
  fi
done
