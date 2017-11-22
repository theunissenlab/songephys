from pydrive.auth import GoogleAuth

import config


def get_auth():
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()
    return gauth
