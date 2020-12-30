#!/bin/bash

helpFunction(){
echo ""
echo "Usage: $0 -name kernelName"
echo -e "\t-name name of kernel in active conda env"
exit 1
}

python3 -m ipykernel install --user --name=$1
