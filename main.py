from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from dadata import Dadata
import numpy as np
import geopy
import geopy.distance
import pandas as pd
import pickle
from io import BytesIO
# import xgboost

app = FastAPI(title='Популярная геолокация',
              description='')

# Секретка
token = "4b8b451fade0ca2e9dd4fbe7981e16824d83d619"
secret = "9d5947f56cea99fb66e83c82d46662ea195fc008"

# Данные
df = pd.read_parquet('df_1-2_new.parquet')

city_params_dict = {
     'Москва':(56.031959, 36.797755, 343),
     'Санкт-Петербург': (60.246734, 29.424817, 220),
     'Новосибирск': (55.197927, 82.740514, 150),
     'Екатеринбург': (56.982824, 60.383625, 144),
     'Казань': (55.919676, 48.846786, 144),
     'Нижний Новгород': (56.420110, 43.719819, 84),
     'Красноярск': (56.133141, 92.652852, 100),
     'Челябинск': (55.318748, 61.241503, 100),
     'Самара': (53.426067, 50.007146, 127),
     'Уфа': (54.959111, 55.765642, 170),
     'Ростов-на-Дону': (47.366032, 39.527174, 80),
     'Краснодар': (45.140525, 38.843852, 100),
     'Омск': (55.140361, 73.095170, 130),
     'Воронеж': (51.816681, 39.013137, 130),
     'Пермь': (58.180438, 55.797528, 170),
     'Волгоград': (48.894814, 44.069950, 170)
}

def findloc(lat: float, lon: float, poly: np.array) -> list:
    """
    Принимает на вход широту и долготу объекта, координатную сетку
    Возвращает индексы квадрата в сетке с наименьшим геометрическим расстоянием от центра квадрата до заданной точки
    """

    assert isinstance(lat, float)
    assert isinstance(lon, float)
    assert isinstance(poly, np.ndarray)

    j = np.argmin(np.abs(poly[0,:,0] - lat))
    k = np.argmin(np.abs(poly[1,0,:] - lon))

    return [j, k]

#
@app.get('/predict_class_by_adress')
def predict_class_by_adress(adress: str):
    '''

    :param adress:
    :return:
    '''

    # подаем токен, секретный код и адрес, для которого хотим получить score в dadata
    dadata = Dadata(token, secret)
    result = dadata.clean('address', adress)

    # дробим на координаты
    latitude = float(result['geo_lat'])
    longitude = float(result['geo_lon'])

    # Длина одного гексагона в метрах
    cell_side = 300
    mapped_objects_dfs = []

    for curr_city in city_params_dict:

        # левый верхний угол (северо-западный)
        north, west, N = city_params_dict[curr_city]

        # начало сетки
        start = geopy.Point(north, west)

        # откладываем сетку на восток и юг
        distance_longitude = geopy.distance.distance(kilometers=(N * cell_side / 1_000))
        distance_latitude = geopy.distance.distance(kilometers=(N * cell_side / 1_000))

        # #считаем расстояние и получаем координаты востока и юга
        destination_east = distance_longitude.destination(point=start, bearing=90)
        destination_south = distance_latitude.destination(point=start, bearing=180)

        east = destination_east.longitude
        south = destination_south.latitude

        if south <= latitude <= north and west <= longitude <= east:
            print(f'Вы попали в город: {curr_city}')

            sn = (south - north) / N
            ew = (east - west) / N

            upper_left = np.mgrid[north:south:sn, west:east:ew]
            upper_right = np.mgrid[north:south:sn, west + ew:east + ew:ew]
            lower_left = np.mgrid[north + sn:south + sn:sn, west:east:ew]
            lower_right = np.mgrid[north + sn:south + sn:sn, west + ew:east + ew:ew]

            grid = pd.DataFrame()

            grid["upper_left_lat"] = upper_left[0].reshape(N * N)
            grid["upper_left_lon"] = upper_left[1].reshape(N * N)

            grid["upper_right_lat"] = upper_right[0].reshape(N * N)
            grid["upper_right_lon"] = upper_right[1].reshape(N * N)

            grid["lower_left_lat"] = lower_left[0].reshape(N * N)
            grid["lower_left_lon"] = lower_left[1].reshape(N * N)

            grid["lower_right_lat"] = lower_right[0].reshape(N * N)
            grid["lower_right_lon"] = lower_right[1].reshape(N * N)

            grid["X_ID"] = np.mgrid[0:N:1, 0:N:1][0].reshape(N * N)
            grid["Y_ID"] = np.mgrid[0:N:1, 0:N:1][1].reshape(N * N)

            grid["cell_center_lon"] = (grid.upper_right_lon - grid.upper_left_lon) / 2 + grid.upper_left_lon
            grid["cell_center_lat"] = (grid.upper_right_lat - grid.lower_right_lat) / 2 + grid.lower_right_lat

            cell_center = np.zeros(shape=(2, N, N))

            cell_center[0] = np.array(grid.cell_center_lat).reshape(N, N)
            cell_center[1] = np.array(grid.cell_center_lon).reshape(N, N)

            hexagon = str(findloc(lat=latitude,
                                  lon=longitude,
                                  poly=cell_center))

            hex_features_vector = df[(df['city'] == curr_city) & (df['cell'] == hexagon)] \
                .drop(['city', 'cell', 'avg_score', 'cnt_atms'], axis=1)


            with open('model_xgb.pkl', 'rb') as file:
                model = pickle.load(file)
                result = model.predict(hex_features_vector)

            return int(result[0])

        else:
            continue

    return 'Вы промахнулись мимо сетки городов, попробуйте снова'


@app.post('/predict_class_by_addresses')
def predict_class_by_addresses(file: UploadFile = File()) -> FileResponse:
    '''
    Функция на вход принимает csv файл,
    где представлен перечень адресов.
    На выходе функция выдает csv файл,
    где для каждого адреса проставлена
    метка класса
    '''

    content = file.file.read()
    test = BytesIO(content)
    test = pd.read_csv(test).to_numpy()
    arr = []

    arr_result = []

    for num in test:
        arr.append(num[0])

    for adress in arr:

        # подаем токен, секретный код и адрес, для которого хотим получить score в dadata
        dadata = Dadata(token, secret)
        result = dadata.clean('address', adress)

        # дробим на координаты
        latitude = float(result['geo_lat'])
        longitude = float(result['geo_lon'])

        # Длина одного гексагона в метрах
        cell_side = 300

        for curr_city in city_params_dict:

            # левый верхний угол (северо-западный)
            north, west, N = city_params_dict[curr_city]

            # начало сетки
            start = geopy.Point(north, west)

            # откладываем сетку на восток и юг
            distance_longitude = geopy.distance.distance(kilometers=(N * cell_side / 1_000))
            distance_latitude = geopy.distance.distance(kilometers=(N * cell_side / 1_000))

            # #считаем расстояние и получаем координаты востока и юга
            destination_east = distance_longitude.destination(point=start, bearing=90)
            destination_south = distance_latitude.destination(point=start, bearing=180)

            east = destination_east.longitude
            south = destination_south.latitude

            if south <= latitude <= north and west <= longitude <= east:
                print(f'Вы попали в город: {curr_city}')

                sn = (south - north) / N
                ew = (east - west) / N

                upper_left = np.mgrid[north:south:sn, west:east:ew]
                upper_right = np.mgrid[north:south:sn, west + ew:east + ew:ew]
                lower_left = np.mgrid[north + sn:south + sn:sn, west:east:ew]
                lower_right = np.mgrid[north + sn:south + sn:sn, west + ew:east + ew:ew]

                grid = pd.DataFrame()

                grid["upper_left_lat"] = upper_left[0].reshape(N * N)
                grid["upper_left_lon"] = upper_left[1].reshape(N * N)

                grid["upper_right_lat"] = upper_right[0].reshape(N * N)
                grid["upper_right_lon"] = upper_right[1].reshape(N * N)

                grid["lower_left_lat"] = lower_left[0].reshape(N * N)
                grid["lower_left_lon"] = lower_left[1].reshape(N * N)

                grid["lower_right_lat"] = lower_right[0].reshape(N * N)
                grid["lower_right_lon"] = lower_right[1].reshape(N * N)

                grid["X_ID"] = np.mgrid[0:N:1, 0:N:1][0].reshape(N * N)
                grid["Y_ID"] = np.mgrid[0:N:1, 0:N:1][1].reshape(N * N)

                grid["cell_center_lon"] = (grid.upper_right_lon - grid.upper_left_lon) / 2 + grid.upper_left_lon
                grid["cell_center_lat"] = (grid.upper_right_lat - grid.lower_right_lat) / 2 + grid.lower_right_lat

                cell_center = np.zeros(shape=(2, N, N))

                cell_center[0] = np.array(grid.cell_center_lat).reshape(N, N)
                cell_center[1] = np.array(grid.cell_center_lon).reshape(N, N)

                hexagon = str(findloc(lat=latitude,
                                      lon=longitude,
                                      poly=cell_center))

                hex_features_vector = df[(df['city'] == curr_city) & (df['cell'] == hexagon)] \
                    .drop(['city', 'cell', 'avg_score', 'cnt_atms'], axis=1)

                with open('model_xgb.pkl', 'rb') as file:
                    model = pickle.load(file)
                    result = model.predict(hex_features_vector)
                    arr_result.append((adress, int(result[0])))

            else:
                continue

    df_res = pd.DataFrame(arr_result, columns=['Адрес', 'Метка класса'])
    df_res.to_csv('result.csv', index=False)
    response = FileResponse(path='result.csv', media_type='text/csv')

    return response