import pickle, os, pgeocode, geopy.distance
from tqdm import tqdm
import multiprocessing as mp
import numpy as np
zipcode_database = pgeocode.Nominatim('us')

def ziplookup_kernel(zipcode):
	zipcode = str(zipcode).zfill(5)
	result = zipcode_database.query_postal_code(zipcode)
	if np.isnan(result.latitude) or np.isnan(result.longitude):
		return None
	return (result.latitude, result.longitude)

def cacheZipcodes():
	zipcodes = list(range(100000))
	pool = mp.Pool(64)
	results = list(tqdm(pool.imap(ziplookup_kernel, zipcodes), total=len(zipcodes)))
	cache = dict()
	for z, r in enumerate(results):
		if r is not None:
			cache[str(z).zfill(5)] = r
	with open("data/preprocessing/zipcodes.pkl", "wb") as file:
		pickle.dump(cache, file)

def distance_2zipcodes(zip1:str, zip2:str) -> float:
	if not os.path.exists("data/preprocessing/zipcodes.pkl"):
		raise Exception("cache file not found")
	with open("data/preprocessing/zipcodes.pkl", "rb") as file:
		cacheZip = pickle.load(file)
	zip1 = str(zip1).zfill(5)
	zip2 = str(zip2).zfill(5)
	if zip1 not in cacheZip or zip2 not in cacheZip:
		return None
	return geopy.distance.geodesic(cacheZip[zip1], cacheZip[zip2]).miles

if __name__=='__main__':
	cacheZipcodes()