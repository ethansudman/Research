import urllib.request as req

def DownloadAntFiles():
    # 4 colonies
    for colony in range(1, 5):
        # 42 days of observations for each
        for day in range(1, 42):
            file = 'ant_mersch_col{c}_day{d}_attribute.graphml'.format(c = str(colony), d = str(day).rjust(2, '0'))
            url = 'https://raw.githubusercontent.com/bansallab/asnr/master/Networks/Insecta/ants_proximity_weighted/' + file

            req.urlretrieve(url, file)