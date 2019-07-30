import datetime


def session_datetime(session_name):
    time_info = session_name[-13:]
    return datetime.datetime(
        year=2000 + int(time_info[:2]),
        month=int(time_info[2:4]),
        day=int(time_info[4:6]),
        hour=int(time_info[7:9]),
        minute=int(time_info[9:11]),
        second=int(time_info[11:13])
    )


def session_time_subtract(session1, session2):
    return (session_datetime(session1) - session_datetime(session2)).seconds


if __name__ == "__main__":
    import sys
    from format.io import load_yaml, save_yaml

    yaml_file = load_yaml(sys.argv[1])

    first_session = yaml_file["sessions"][0]

    for session in yaml_file["sessions"]:
        print("correcting {}".format(session))
        t = session_time_subtract(session["id"], first_session["id"])
        print("corrected to {}".format(t))
        session["t"] = float(t)
    
    save_yaml(sys.argv[1], yaml_file)
