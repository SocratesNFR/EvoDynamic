""" Utils """

def progressbar(current, total):
  # Based on: https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console/30740258
  percentage = 100*current/total
  print("\r[%-100s] %.2f%%" % ('='*int(percentage), percentage), end='\r')
  if current == total:
    print()

def progressbar_loss(current, total, loss):
  # Based on: https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console/30740258
  percentage = 100*current/total
  print("\r[%-100s] %.2f%%. Loss: %.5f" % ('='*int(percentage), percentage, loss), end='\r')
  if current == total:
    print()