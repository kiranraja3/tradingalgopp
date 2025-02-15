def is_valid_ip(ip_string):
    #creates segments by splitting each by period
    segments = ip_string.split('.')

    #IPs can only have 4 segments
    if len(segments) != 4:
        return False

    for segment in segments:
        try:
            #converts each segment into integer
            num = int(segment)
            if num < 0 or num > 255:
                return False
        except ValueError:
            #if converting integer raises error, then non-integer characters are present
            return False

    return True

def get_valid_ip(ip_str):

    valid_ips = []
    for ip in ip_str.split():
        #cycles through each ip entered and checks if valid, if so appending to initialized list
        if is_valid_ip(ip):
            valid_ips.append(ip)

    return valid_ips

# Example usage:
ip_input = input("Enter IPs separated by spaces: ")
valid_ips = get_valid_ip(ip_input)

if len(valid_ips) == 0:
    #if no valid ips or no ips were entered then user is informed
    print("No valid Ips")
else:
    #prints valid ips
    print("valid IPs are:", valid_ips)

