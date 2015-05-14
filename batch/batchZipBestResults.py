
from common import output

best_res_files = ['models_lj2m_dev3/best_results_[0, 1, 2, 3, 4, 5, 6, 7, 8, 9].pkl', 
                    'models_lj2m_dev3/best_results_[10, 11, 12, 13, 14, 15, 16, 17, 18, 19].pkl', 
                    'models_lj2m_dev3/best_results_[20, 21, 22, 23, 24, 25, 26, 27, 28, 29].pkl', 
                    'models_lj2m_dev3/best_results_[30, 31, 32, 33, 34, 35, 36, 37, 38, 39].pkl']

res = output.PredictionResult(best_res_files)

output_filename = 'zipped_best_results.pkl'
res.dump_results(output_filename)

summary_filename = 'best_result_summary.csv'
res.dump_summary(summary_filename)
