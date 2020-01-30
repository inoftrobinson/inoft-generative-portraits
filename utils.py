def week_number_of_month(self, date_value):
    # Source : https://www.mytecbits.com/internet/python/week-number-of-month
    return date_value.isocalendar()[1] - date_value.replace(day=1).isocalendar()[1] + 1
