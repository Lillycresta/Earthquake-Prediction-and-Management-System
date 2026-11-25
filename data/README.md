# Dataset

Source: USGS Earthquake Catalog

Download scope:
- Global earthquakes
- 2016-01-01 through 2025-12-31
- Magnitude 4.5 or greater
- CSV format

The raw file is preserved in `data/raw/`. Run `python preprocess_train.py` to clean the data and train the three configured classification models.

No synthetic earthquake records are used.
