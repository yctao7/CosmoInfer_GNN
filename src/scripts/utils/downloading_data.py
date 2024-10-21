import requests
from paths import data, datapath
import os

# Modify these parameters to decide what simulations to download
sims = ["IllustrisTNG", "SIMBA", "Astrid", "EAGLE"]
indexes = range(1000)

"""Download the dataset from Flatiron Institute.
e.g. link: https://users.flatironinstitute.org/~camels/FOF_Subfind/IllustrisTNG/LH/LH_567/fof_subhalo_tab_005.hdf5 

Args:
    sims (lst): strings of simulations, "IllustrisTNG" or "SIMBA"
    indexes (ndarray): Indexes of subfind data from LH dataset to be downloaded.
"""

# Create the data directory if it doesn't exist
data.mkdir(parents=True, exist_ok=True)

destination = datapath
url_prefix = "https://users.flatironinstitute.org/~camels/FOF_Subfind/"
suffix = "groups_090.hdf5"

seeds = {
    "IllustrisTNG": "https://github.com/franciscovillaescusa/CAMELS/blob/master/docs/params/IllustrisTNG/CosmoAstroSeed_IllustrisTNG_L25n256_LH.txt",
    "SIMBA": "https://github.com/franciscovillaescusa/CAMELS/blob/master/docs/params/SIMBA/CosmoAstroSeed_SIMBA_L25n256_LH.txt",
    "Astrid": "https://github.com/franciscovillaescusa/CAMELS/blob/master/docs/params/Astrid/CosmoAstroSeed_Astrid_L25n256_LH.txt",
    "EAGLE": "https://github.com/franciscovillaescusa/CAMELS/blob/master/docs/params/EAGLE/CosmoAstroSeed_SwiftEAGLE_L25n256_LH.txt"
}

for sim in sims:
    if not os.path.exists(destination + f"CosmoAstroSeed_params_{sim}.txt"):
        with open(destination + f"CosmoAstroSeed_params_{sim}.txt", "wb") as f:
            f.write(requests.get(seeds[sim]+"?raw=true").content)
    for i in indexes:
        url = url_prefix + sim + "/LH/LH_" + str(i) + "/" + suffix
        name = destination + sim + "_LH_" + str(i) + "_" + suffix
        if not os.path.exists(name):
            r = requests.get(url)
            f = open(name, 'wb')
            f.write(r.content)
            print(f"File downloaded for {sim} set {i}")
            f.close()
