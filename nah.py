import NeuroticAdHominem as nah
from sys import argv

if(argv[1] == "train"):
    nah.train()
elif(argv[1] == "launch"):
    nah.launch()
else:
    print("Please enter train or launch.")

