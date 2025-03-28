from SoccerNet.Downloader import SoccerNetDownloader

local_dir = "/scratch/eecs545w25_class_root/eecs545w25_class/highlights/SoccerNet"
mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory=local_dir)
mySoccerNetDownloader.password = "s0cc3rn3t"
mySoccerNetDownloader.downloadGames(files=["1_720p.mkv", "2_720p.mkv", "video.ini", "Labels-v2.json"], split=["train","valid","test"])
