from geopkg.io import las_io, tops_io, core_io

logs_path = r'C:\Users\RY47284\OneDrive - Repsol\Codes\PyThings\Jupyters\PyWellData\PermPred\Logs'
tops_path = r'C:\Users\RY47284\OneDrive - Repsol\Codes\PyThings\Jupyters\PyWellData\PermPred\Well_tops.txt'
core_path = r'C:\Users\RY47284\OneDrive - Repsol\Codes\PyThings\Jupyters\PyWellData\PermPred\core_data.ascii.txt'

logs = las_io.las_multi(logs_path)
logs.rename(mapper={'DEPT':'MD'}, axis=1, inplace=True)

tops = tops_io.read_tops(tops_path, colnames=['Well', 'Surface', 'MD'])

ccal = core_io.read_core(core_path)

#logs = tops_io.merge_tops(logs, tops, 'Surface', 'Well', 'MD', ['Overbuden', 'Base'])
print(logs.head())