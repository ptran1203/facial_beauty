# import matplotlib.pyplot as plt
import pandas as pd
import urllib.request
def plot_ds_histogram():
    path = 'dataset/all_labels.csv'
    df = pd.read_csv(path)
    labels = df['labels']

    print(labels)

# plot_ds_histogram()

def download():
	paths = [
	'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ2gqkpgKhpLSH0AZEVnZ7Iyc9Xu13ni-cese0KSGEyAbDTgvJGPw&s',
	'https://icdn.dantri.com.vn/thumb_w/640/2019/02/23/co-gai-viet-dep-tua-hot-girl-co-guong-mat-dep-nhat-han-quoc-1-1550894038111.jpg',
	'https://cdn.pose.com.vn/legacy/images/baiviet/201612//35246_2_2.jpg',
	'https://vcdn-vnexpress.vnecdn.net/2019/09/04/1-7011-1567586536.jpg',
	'http://tainangviet.vn//source/image/StartUp/Nhung-co-gai-xinh-dep-khoi-nghiep.jpg',
	'https://nld.mediacdn.vn/2019/5/31/bacgiang-15592702415461612633728.jpg',
	'https://6.viki.io/image/6e9e4b1da4a4489b940b1a7291db0240.jpeg?s=900x600&e=t',
	'https://imgix.bustle.com/uploads/image/2019/7/15/bb4c7b65-b250-4cba-8c4b-bd0bfcbca610-290ab36a-6b94-483b-a61f-4_b0af5b9640f-getty-1138777447.jpg',
	'https://baoquocte.vn/stores/news_dataimages/nguyennga/042017/28/14/140601_2804_emmawatson.jpg',
	'https://www.telegraph.co.uk/content/dam/news/2019/09/29/TELEMMGLPICT000211354381_trans_NvBQzQNjv4BqrpfQw2hJyG_yckwxPAr0go9KzD8cVu9iguqnaKUswZA.jpeg?imwidth=450',
	]

	for i,path in enumerate(paths):
		data = urllib.request.urlretrieve(path, 'img' + str(i) + '.png')
		print(data)


def dataset_distribute():
    df = pd.read_csv('dataset/all_labels.csv')
    group_ = {
        'AF': [],
        'AM': [],
        'CF': [],
        'CM': [],
    }
    # bins = numpy.linspace(1, 5, 0.5)

    def fgroup(filename):
        for g in group_.keys():
          if g in filename:
            return g
        return False

    labels = df['score']
    files = df['filename']
    for i in range(len(labels)):
        filename = files[i]
        label = labels[i]
        gname = fgroup(filename)
        group_[gname].append(label)
    
    for gr, vals in group_.items():
        print(gr, len(vals))
dataset_distribute()
