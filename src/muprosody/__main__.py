import sys
from muprosody import (
    run_myspsolution_praat_file,
    run_mltrnl_praat_file,
    score_cefr_level,
)


def main():
    if len(sys.argv) < 3:
        print("Usage: <function_name> <audio_file_path>")
        sys.exit(1)

    function_name = sys.argv[1]
    audio_file_path = sys.argv[2]

    if function_name == "run_myspsolution_praat_file":
        run_myspsolution_praat_file(audio_file_path)
    elif function_name == "run_mltrnl_praat_file":
        run_mltrnl_praat_file(audio_file_path)
    elif function_name == "score_cefr_level":
        score_cefr_level(audio_file_path)
    else:
        print("Invalid function name")
        sys.exit(1)


if __name__ == "__main__":
    main()
