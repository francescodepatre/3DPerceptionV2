import requests

class GPS:
    def __init__(self):
        self

    def get_location():
        try:
            # Richiesta all'API di ipinfo.io
            response = requests.get('https://ipinfo.io/json')
            
            # Controlla se la richiesta ha avuto successo
            if response.status_code == 200:
                data = response.json()
                # Estrai latitudine e longitudine
                loc = data.get('loc', '0,0').split(',')
                latitude = loc[0]
                longitude = loc[1]
                return latitude, longitude
            else:
                print("Errore nella richiesta:", response.status_code)
                return None, None
        except Exception as e:
            print("Errore:", e)
            return None, None

'''
# Usa la funzione
latitude, longitude = get_location()
if latitude and longitude:
    print(f"La tua posizione: Latitudine = {latitude}, Longitudine = {longitude}")
'''