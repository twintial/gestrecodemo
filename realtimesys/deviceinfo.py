import pyaudio

p = pyaudio.PyAudio()


def get_drivers():
    driver_dicts = []
    for i in range(p.get_host_api_count()):
        driver_dicts.append(p.get_host_api_info_by_index(i))
    return driver_dicts


def get_devices(driver_dicts, kind='Input', driver='ASIO'):
    driver_dict = {}
    devices_name = []
    for d in driver_dicts:
        if d['name'] == driver:
            driver_dict = d
            break
    for i in range(driver_dict['deviceCount']):
        device_dict = p.get_device_info_by_host_api_device_index(driver_dict['index'], i)
        if kind and device_dict[f'max{kind}Channels'] > 0:
            devices_name.append({'name': device_dict['name'],
                                 'id': device_dict['index'],
                                 'channels': device_dict[f'max{kind}Channels']})
    return devices_name
