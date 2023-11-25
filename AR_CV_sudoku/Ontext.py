def get_vars(out_corners_check, solved, bad_read,time_on_corners,seen,solved_text):
    if seen and solved and not bad_read and not out_corners_check:
        return solved_text, (285, 30),(0, 255, 0)

    return '', (320, 30) ,(0, 0, 255)

def dots(time_out_corners):
    multiplier = int(time_out_corners // 1)
    nasobek = int(multiplier/5) +1
    if multiplier > (5 * nasobek):
        nasobek += 1
    tecky = 5 + multiplier - (5 * nasobek)
    return '.' * tecky
