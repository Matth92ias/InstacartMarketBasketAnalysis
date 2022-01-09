import pandas as pd
import os

def ImportOrders(data_dir,small_sample):
   import_dir = os.path.join(data_dir,"01_raw")
   files = ["order_products__prior","order_products__train","orders"] #
   if small_sample == True:
       files = [file + '_sample' for file in files]
   file_tuple = tuple(pd.read_csv(os.path.join(import_dir,file + ".csv")) for file in files)
   return file_tuple

def ImportAddInfos(data_dir):
    import_dir = os.path.join(data_dir,"01_raw")
    files = ["aisles","departments","products"]
    file_tuple = tuple(pd.read_csv(os.path.join(import_dir,file + ".csv")) for file in files)
    return file_tuple

def SaveSampleOfOrders(data_dir,order_products__prior, order_products__train, orders,user_number = 100):
    # filter by users and save sample of files into data directory
    import_dir = os.path.join(data_dir,"01_raw")

    orders = orders.loc[orders.user_id <= user_number,:]
    order_products__train = order_products__train.merge(orders.loc[:, ['order_id','user_id'] ],how="inner",on = "order_id").drop(["user_id"],axis=1)
    order_products__prior = order_products__prior.merge(orders.loc[:, ['order_id','user_id'] ],how="inner",on = "order_id").drop(["user_id"],axis=1)

    orders.to_csv(os.path.join(import_dir,'orders_sample' + ".csv"),index=False)
    order_products__train.to_csv(os.path.join(import_dir,'order_products__train_sample' + ".csv"),index=False)
    order_products__prior.to_csv(os.path.join(import_dir,'order_products__prior_sample' + ".csv"),index=False)

def SampleSaveFlatfile(flatfile,size,data_dir):
    unique_ids = flatfile["user_id"].unique()
    unique_ids = np.random.choice(unique_ids,size)
    flatfile_sample = flatfile.loc[flatfile["user_id"].isin(unique_ids),:].reset_index(drop=True)
    flatfile_sample.to_csv(os.path.join(data_dir,"02_intermediate/flatfile_sample.csv"),index=False)
