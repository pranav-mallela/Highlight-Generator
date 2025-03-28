LEAGUE="*"
SEASON="*"
SOCCERNET_ROOT="/scratch/eecs545w25_class_root/eecs545w25_class/highlights/SoccerNet"

while [[ $# -gt 0 ]]; do
	case $1 in
		-l|--league)
			LEAGUE="$2"
			shift
			shift
            ;;
        -s|--season)
            SEASON="$2"
            shift
            shift
            ;;
        *)
            echo "Unknown option/argument"
            exit 1
            ;;
    esac
done

echo "Loading ffmpeg module...";
ml ffmpeg;
echo "Loaded ffmpeg";
echo;

find $SOCCERNET_ROOT/$LEAGUE/$SEASON -name "*_720p.mkv" -exec bash -c \
	'video_path=$1;
	echo "Processing video: $video_path";
	part_path="$(dirname -- "$video_path")/chunked";
	mkdir -p "$part_path";
	chunk_prefix=$(basename -- "$video_path" | cut -d "_" -f 1);
	chunk_name="${chunk_prefix}_%03d.mkv";
	ffmpeg \
		-i "$video_path" \
		-loglevel 8  \
		-c copy -map 0 \
		-segment_time 00:05:00 \
		-f segment \
		-reset_timestamps 1 \
		"$part_path/$chunk_name";
	chunk_count=$(ls "$part_path" | grep "${chunk_prefix}_" | wc -l);
	echo -e "\tSuccessfully generated $chunk_count chunks";
	echo
	' _ {} \;

