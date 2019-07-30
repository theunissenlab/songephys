from pydrive.auth import GoogleAuth


def get_auth():
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()
    return gauth
