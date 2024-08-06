-------------------------------------------------------------------------------
-- Synchronous RAM with generic memory and data sizes
-------------------------------------------------------------------------------

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity ram is
    generic(ADDRESS_BITS: positive;
            DATA_LENGTH:  positive);
    port(-- Control signals
         Clk: in std_logic;
         We:  in std_logic;
         Re:  in std_logic;
         
         -- Input signals
         Addr: in std_logic_vector (ADDRESS_BITS - 1 downto 0);
         Din:  in std_logic_vector (DATA_LENGTH - 1 downto 0);
         
         -- Output
         Dout: out std_logic_vector (DATA_LENGTH - 1 downto 0));
end ram;

architecture Behavioral of ram is

type MemoryBank is array(0 to 2**ADDRESS_BITS - 1)
                   of std_logic_vector(DATA_LENGTH - 1 downto 0);
signal bank: MemoryBank;

begin
process (Clk)
begin
    if rising_edge(Clk) then
        if (We = '1') then
            -- Write into Addr
            bank(to_integer(unsigned(Addr))) <= Din;
        end if;
        if (Re = '1') then
            -- Read from Addr
            Dout <= bank(to_integer(unsigned(Addr)));
        else
            Dout <= (others => '0');
        end if;
    end if;
end process;
end Behavioral;

