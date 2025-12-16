from mockup_algo import run as mockup_run

def retrieve_and_compute(df, genres_dict, query_artist, query_track, query_album, algorithms, n):
    output_all = {}
    for algo in algorithms:
        if algo == "Mockup":
            output = mockup_run(df, genres_dict, query_artist, query_track, query_album, n)
        elif algo == "Mockup2":
            output = mockup_run(df, genres_dict, query_artist, query_track, query_album, n)
        else:
            output = {"results": [], "metrics": {}}
        output_all[algo] = output
    return output_all
