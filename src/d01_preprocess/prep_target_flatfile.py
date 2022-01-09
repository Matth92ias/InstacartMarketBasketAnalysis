import pandas as pd

def join_order_prod_files(order_products__prior, order_products__train, orders):
    '''
    Join both order_prdoduct files and add additional information

    :param order_products__prior: product order prior to train
    :param order_products__train: product orders in train
    :param orders: all orders

    :return: concated dataframe containing both order_products dataframes
    '''
    # union all all info about which product in which order
    order_products__prior['eval_set'] = 'prior'
    order_products__train['eval_set'] = 'train'
    order_products_gesamt = pd.concat([order_products__prior, order_products__train])
    # add info about user_id and order_nr
    order_products_gesamt = order_products_gesamt.merge(orders.loc[:, ['order_id', 'user_id', 'order_number']],
                                                        how="inner", on="order_id")
    return order_products_gesamt


def prep_flatfile(order_product_gesamt, orders, which_set):
    '''
    get all previously ordered items to create flatfile

    :param order_product_gesamt:
    :param orders:
    :param which_set:

    :return: flatfile
    '''
    # get all items previously ordered items - in train only check items which were ordered prior
    # (automatically excludes products which were ordered in train for the first time)
    if which_set == 'train':
        order_product_gesamt = order_product_gesamt.loc[order_product_gesamt.eval_set != 'train']
    flatfile = order_product_gesamt.loc[:, ['user_id', 'product_id']].drop_duplicates().reset_index(drop=True)

    # get all orders from the specific set - some users are only in train or test (or neither?)
    rel_orders = orders.loc[orders.eval_set == which_set, ['user_id', 'order_id', 'order_number']].rename(
        columns={'order_number': 'ff_order_number', 'order_id': 'ff_order_id'})

    flatfile = flatfile.merge(rel_orders, how='inner', on='user_id')

    return flatfile


def prep_target(train, order_product_gesamt):
    '''
    Prepare the target which is 1 if a product is in the flatfile order and 0 otherwise
    :param train:
    :param order_product_gesamt:
    :return:
    '''
    # get all orders of user
    prod_in_train = order_product_gesamt.loc[
        order_product_gesamt.eval_set == 'train', ['order_id', 'product_id', 'eval_set']]

    train = train.merge(prod_in_train, left_on=['ff_order_id', 'product_id'], right_on=['order_id', 'product_id'],
                        how='left')

    train['target'] = train['eval_set'].apply(lambda x: 0 if pd.isnull(x) else 1)

    train = train.drop(columns=['order_id', 'eval_set'])

    return train
