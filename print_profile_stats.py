import pstats
from pstats import SortKey

p = pstats.Stats('execute_profile')
# p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(30)
p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_callers(30)
