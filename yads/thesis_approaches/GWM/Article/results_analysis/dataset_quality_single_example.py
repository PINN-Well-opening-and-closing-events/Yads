{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "ename": "ModuleNotFoundError",
     "evalue": "No module named 'yads'",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mModuleNotFoundError\u001b[0m                       Traceback (most recent call last)",
      "Cell \u001b[0;32mIn[1], line 11\u001b[0m\n\u001b[1;32m      8\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01msys\u001b[39;00m\n\u001b[1;32m     10\u001b[0m sys\u001b[38;5;241m.\u001b[39mpath\u001b[38;5;241m.\u001b[39mappend(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m/home/AD.NORCERESEARCH.NO/anle/Yads\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n\u001b[0;32m---> 11\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01myads\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mmesh\u001b[39;00m \u001b[38;5;28;01mas\u001b[39;00m \u001b[38;5;21;01mym\u001b[39;00m\n\u001b[1;32m     12\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01myads\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mmesh\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mutils\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m load_json\n",
      "\u001b[0;31mModuleNotFoundError\u001b[0m: No module named 'yads'"
     ]
    }
   ],
   "source": [
    "from scipy.stats import wasserstein_distance\n",
    "import pandas as pd\n",
    "from ast import literal_eval\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from joblib import Parallel, delayed\n",
    "\n",
    "import sys\n",
    "\n",
    "sys.path.append(\"/home/AD.NORCERESEARCH.NO/anle/Yads\")\n",
    "import yads.mesh as ym\n",
    "from yads.mesh.utils import load_json"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def compute_distance(sample_1, sample_2):\n",
    "    # normalize\n",
    "    sample_1 = (sample_1 - sample_1.min()) / (sample_1.max() - sample_1.min())\n",
    "    sample_2 = (sample_2 - sample_2.min()) / (sample_2.max() - sample_2.min())\n",
    "    return wasserstein_distance(sample_1, sample_2)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Yads",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
