
from common import output

best_res_files = ['aaa.pkl', 'bbb.pkl', 'ccc.pkl', 'ddd.pkl']

res = output.PredictionResult(best_res_files)

output_filename = 'zipped_best_results.pkl'
res.dump_results(output_filename)

summary_filename = 'best_result_summary.csv'
res.dump_summary(summary_filename)
