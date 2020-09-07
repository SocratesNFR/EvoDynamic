""" Utils """

def progressbar(current, total):
  # Based on: https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console/30740258
  percentage = 100*current/total
  print("\r[%-50s] %.2f%%" % ('='*int(percentage//2), percentage), end='\r')
  if current == total:
    print()

def progressbar_loss(current, total, loss):
  # Based on: https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console/30740258
  percentage = 100*current/total
  print("\r[%-50s] %.2f%%. Loss: %.5f" % ('='*int(percentage//2), percentage, loss), end='\r')
  if current == total:
    print()

def progressbar_loss_accu(current, total, loss, accu):
  # Based on: https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console/30740258
  percentage = 100*current/total
  print("\r[%-50s] %.2f%%. Loss: %.5f. Accuracy: %.5f" % ('='*int(percentage//2), percentage, loss, accu), end='\r')
  if current == total:
    print()