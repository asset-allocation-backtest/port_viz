import porfolio_vis as pv

# pv.action(data_list=['002200','005930','005380','000660'],
#           ratio_list=[0.1,0.4,0.2,0.3],
#           country='kr',
#           window_hold='Q',
#           rebalancing_date=-5)

self = pv.action(data_list=['SPY','ACWI','IEI','IEF'],
                ratio_list=[0.1,0.4,0.2,0.3],
                country='us',
                window_hold='M',
                rebalancing_date=-1)
