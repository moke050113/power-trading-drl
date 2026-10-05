class SpotMarket:
    """
    简化的现货电力交易市场模型
    """

    def __init__(self, price):
        """
        初始化现货市场

        参数：
            price：现货电价，单位：元/MWh
        """
        self.price = price

    def trade(self, quantity):
        """
        进行现货购电交易

        参数：
            quantity：购买电量，单位：MWh

        返回：
            cost：购电成本，单位：元
        """

        if quantity < 0:
            raise ValueError("购买电量不能小于0")

        cost = quantity * self.price

        return cost