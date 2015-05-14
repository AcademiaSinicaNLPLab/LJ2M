
from common import output

best_res_files = ['aaa.pkl', 'bbb.pkl', 'ccc.pkl', 'ddd.pkl']

res = output.PredictionResult(best_res_files)

output_filename = 'zipped_best_results.pkl'
res.dump_summary(output_filename)
