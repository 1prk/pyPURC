#Our estimator for the PURC model requires multiple observations in each OD pair. Because common ODs are very rare in
#a large network, we follow (6) and trim the observed trips such that the trimmed trips share common ODs.
import h3
import csv
import config.params as params

class OdPreparation:
    def __init__(self):
        self.odDir = params.odDir

    def trim_od_data(self):
        cache = {}
        def get_latlng(h3_index):
            if h3_index not in cache:
                cache[h3_index] = h3.cell_to_latlng(h3_index)
            return cache[h3_index]

        aggregated_ods = []
        h3_metadata = set()

        with open(self.odDir, 'r') as f:
            od_data = csv.reader(f, delimiter=';')
            headers = next(od_data)
            for row in od_data:
                trip_id = row[1]
                start_lat = float(row[10])
                start_lon = float(row[11])
                end_lat = float(row[12])
                end_lon = float(row[13])
                start_h3 = h3.latlng_to_cell(start_lat, start_lon, 9)
                end_h3 = h3.latlng_to_cell(end_lat, end_lon, 9)  # korrigiert
                aggregated_ods.append({
                    'trip_id': trip_id,
                    'start_h3': start_h3,
                    'end_h3': end_h3
                })
                h3_metadata.add((start_h3, get_latlng(start_h3)))
                h3_metadata.add((end_h3, get_latlng(end_h3)))
        return aggregated_ods, h3_metadata

if __name__ == '__main__':
    odPrep = OdPreparation()
    ods, h3_metadata = odPrep.trim_od_data()
    print('lol')
