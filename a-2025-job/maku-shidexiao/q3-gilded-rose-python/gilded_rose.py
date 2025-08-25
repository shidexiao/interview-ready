# -*- coding: utf-8 -*-


class GildedRose(object):

    def __init__(self, items):
        self.items = items

    def update_quality(self):
        for item in self.items:
            self._update_item_quality(item)

    def _update_item_quality(self, item):
        if item.name == "Sulfuras, Hand of Ragnaros":
            return
        item.sell_in = item.sell_in - 1

        if item.name == "Aged Brie":
            self._update_aged_brie(item)
        elif item.name == "Backstage passes to a TAFKAL80ETC concert":
            self._update_backstage_passes(item)
        elif "Conjured" in item.name:
            self._update_conjured_item(item)
        else:
            self._update_normal_item(item)

        item.quality = max(0, min(50, item.quality))

    def _update_aged_brie(self, item):
        item.quality += 1

    def _update_backstage_passes(self, item):
        if item.sell_in < 0:
            item.quality = 0
        elif item.sell_in <= 5:
            item.quality += 3
        elif item.sell_in <= 10:
            item.quality += 2
        else:
            item.quality += 1

    def _update_conjured_item(self, item):
        degrade = 2 if item.sell_in >= 0 else 4
        item.quality -= degrade

    def _update_normal_item(self, item):
        degrade = 1 if item.sell_in >= 0 else 2
        item.quality -= degrade


class Item:
    def __init__(self, name, sell_in, quality):
        self.name = name
        self.sell_in = sell_in
        self.quality = quality

    def __repr__(self):
        return "%s, %s, %s" % (self.name, self.sell_in, self.quality)
