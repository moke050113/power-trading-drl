class LongTermMarket:
    """
    简化的中长期电力交易市场模型
    """

    def __init__(self, price):
        """
        初始化中长期市场

        参数：
            price：中长期电价，单位：元/MWh
        """
        self.price = price

    def trade(self, quantity):
        """
        进行中长期购电交易

        参数：
            quantity：购买电量，单位：MWh

        返回：
            cost：购电成本，单位：元
        """

        if quantity < 0:
            raise ValueError("购买电量不能小于0")

        cost = quantity * self.price

        return cost